# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod
from solo.methods.simclr import SimCLR


class Distill(BaseMethod):
    def __init__(self, proj_output_dim: int, proj_hidden_dim: int, temperature: float, **kwargs):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(**kwargs)

        self.temperature = temperature

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        self.target_network = SimCLR(proj_output_dim, proj_hidden_dim, temperature, **kwargs)
        self.target_network.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(Distill, Distill).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats = out["feats"]
        z = torch.cat([self.projector(f) for f in feats])

        out = self.target_network.base_training_step(batch, batch_idx)
        feats = out["feats"]
        z_target = torch.cat([self.projector(f) for f in feats])

        nce_loss = distill_loss(
            z, z_target.detach(), 
            temperature=self.temperature,
        )
        # nce_loss = - F.cosine_similarity(z, z_target.detach()).mean()

        self.log("train_distill_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss

from solo.utils.misc import gather, get_rank
import torch.nn.functional as F

# def distill_loss(z, z_target, temperature):
#     z = F.normalize(z, dim=-1)
#     id_mask = 1 - torch.eye(z.size(0), dtype=torch.float, device=z.device)
#     sim = torch.einsum("if, jf -> ij", z, z) / temperature * id_mask
#     z_target = F.normalize(z_target, dim=-1)
#     sim_tareget = torch.einsum("if, jf -> ij", z, z) / temperature * id_mask

#     targets = F.softmax(sim_tareget, dim=1)
#     inputs = F.log_softmax(sim, dim=1)
#     return F.kl_div(inputs, targets, reduction='batchmean')

def distill_loss(z, z_target, temperature):
    z1, z2 = z.view(2, -1, z.size(1))
    z1t, z2t = z_target.view(2, -1, z.size(1))

    return kl(z1, z2t, temperature) + kl(z2, z1t, temperature)
    

def kl(inputs, targets, tempearature):
    targets = F.softmax(targets / tempearature, dim=1)
    inputs = F.log_softmax(inputs, dim=1)
    return F.kl_div(inputs, targets, reduction='batchmean')


    # z = F.normalize(z, dim=-1)
    # id_mask = 1 - torch.eye(z.size(0), dtype=torch.float, device=z.device)
    # sim = torch.einsum("if, jf -> ij", z, z) / temperature * id_mask
    # z_target = F.normalize(z_target, dim=-1)
    # sim_tareget = torch.einsum("if, jf -> ij", z, z) / temperature * id_mask

    # targets = F.softmax(sim_tareget, dim=1)
    # inputs = F.log_softmax(sim, dim=1)
    # return F.kl_div(inputs, targets, reduction='batchmean')

