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
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseMethod


class Rot(BaseMethod):
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

        self.rot_classifier = nn.Linear(proj_output_dim, 4)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(Rot, Rot).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("rot")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--temperature", type=float, default=0.1)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        # extra_learnable_params: List[dict] = [
        #     {"params": self.projector.parameters()},
        #     {"params": self.predictor.parameters(), "static_lr": True},
        # ]
        return super().learnable_params

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """
        return self.base_forward(*args, **kwargs)


    def base_forward(self, X: torch.Tensor) -> Dict:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        feats = self.backbone(X)
        logits = self.classifier(feats)
        return {
            "logits": logits,
            "feats": feats,
        }

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimSiam reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss
        """

        # out = super().training_step(batch, batch_idx)
        # class_loss = out["loss"]
        _, x, targets = batch
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        x = x[0]
        angles = torch.randint(4, (x.size(0),), dtype=torch.long, device=x.device)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # for m
            ims = []
            for im, ang in zip(x, angles):
                ims.append(torch.rot90(im, ang, [1,2]))
            x = torch.stack(ims)
        # import pdb; pdb.set_trace()
        feats = self.backbone(x)
        z = self.projector(feats)
        rot_out = self.rot_classifier(z)
        rot_loss = F.cross_entropy(rot_out, angles)
        self.log("train_rot_loss", rot_loss, on_epoch=True, sync_dist=True)

        return rot_loss + class_loss
