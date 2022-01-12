rate = '0.60'
dataset = 'cifar10'
poison_method = 'zoo-simclr'
# poison_method = 'clb'
import time
out_dir = '/data/yfwang/solo-learn-outputs'

def sweep_poison_rate(args):
    i = 0
    # for dataset in ['cifar10', 'cifar100']:
    for dataset in ['cifar10']:
        # for rate in '0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'.split(' '):
        for rate in '0.90 1.00 1.10 1.20'.split(' '):
            gpu = args.gpus[i]
            print(rate)
            os.system(f"""
            for file in /data/yfwang/solo-learn/poison_datasets/{dataset}/zoo-simclr/gaussian_noise/{dataset}_zoo-simclr_rate_{rate}_*.pt
            do 
            # echo ${{file}}
            CUDA_VISIBLE_DEVICES={gpu} sh simclr.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/{dataset} " &
            done
            """
            )
            i += 1
            if i >= len(args.gpus):
                return

def sweep_pretrain_method(args):
    i = 0
    # for dataset in ['cifar10', 'cifar100']:
    # for method in 'dino'.split(' '):
    dataset = 'cifar10'
    for method in ['supcon', 'mocov2plus', 'byol', 'simsiam']:
    # for method in ['simclr', 'sup', 'supcon', 'mocov2plus', 'byol', 'simsiam', 'swav', 'dino', 'barlow']:
        # for rate in '0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'.split(' '):
        gpu = args.gpus[i]
        print(rate)

        os.system(f"""
        for file in {out_dir}/poison_datasets/{dataset}/{poison_method}/gaussian_noise/{dataset}_{poison_method}_rate_{rate}_*.pt
        do 
        # echo ${{file}}, {method}, {dataset}, {out_dir}
        CUDA_VISIBLE_DEVICES={gpu} sh {method}.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir {out_dir}/pretrain/{dataset} " &
        done
        """
        )
        i += 1
        if i >= len(args.gpus):
            return
        time.sleep(1)


def sweep_eval(args):
    i = 0
    for dataset in ['cifar10', 'cifar100']:
    # for method in 'dino'.split(' '):
        for method in ['sup','simclr']:
            for poison_method in ['zoo-simclr', 'clb']:
                # for apply_method in ['use_poison', 'eval_poison']:
                for apply_method in ['eval_poison']:                
            # for rate in '0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'.split(' '):
                    gpu = args.gpus[i]
                    print(rate)

                    os.system(f"""
                    for file in {out_dir}/poison_datasets/{dataset}/{poison_method}/gaussian_noise/{dataset}_{poison_method}_rate_{rate}_*.pt
                    do 
                    # echo ${{file}}, {method}
                    CUDA_VISIBLE_DEVICES={gpu} sh {method}.sh {dataset} " --poison_data ${{file}}  --{apply_method} --checkpoint_dir {out_dir}/pretrain/{dataset} " &
                    done
                    """
                    )
                    i += 1
                    if i >= len(args.gpus):
                        return
                    time.sleep(1)

def sweep_cifar100(args):
    i = 0
    for dataset in ['cifar100']:
    # for method in 'dino'.split(' '):
        for method in ['sup','simclr']:
            for poison_method in ['zoo-simclr', 'clb']:
                # for apply_method in ['use_poison', 'eval_poison']:
                for apply_method in ['use_poison']:                
            # for rate in '0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'.split(' '):
                    gpu = args.gpus[i]
                    print(rate)

                    os.system(f"""
                    for file in {out_dir}/poison_datasets/{dataset}/{poison_method}/gaussian_noise/{dataset}_{poison_method}_rate_{rate}_*.pt
                    do 
                    # echo ${{file}}, {method}
                    CUDA_VISIBLE_DEVICES={gpu} sh {method}.sh {dataset} " --poison_data ${{file}}  --{apply_method} --checkpoint_dir {out_dir}/pretrain/{dataset} " &
                    done
                    """
                    )
                    i += 1
                    if i >= len(args.gpus):
                        return
                    time.sleep(1)


def sweep_poison_pretrain_method(args):
    i = 0
    # for dataset in ['cifar10', 'cifar100']:
    # for method in 'dino'.split(' '):
    for method in ['supcon', 'mocov2plus', 'byol', 'simsiam', 'swav', 'dino', 'barlow']:
        poison_method = 'zoo-' + method
        # for rate in '0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'.split(' '):
        gpu = args.gpus[i]
        print(rate)

        os.system(f"""
        for file in {out_dir}/poison_datasets/{dataset}/{poison_method}/gaussian_noise/{dataset}_{poison_method}_rate_{rate}_*.pt
        do 
        echo ${{file}}, {method}
        # CUDA_VISIBLE_DEVICES={gpu} sh {method}.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir {out_dir}/pretrain/{dataset} " &
        CUDA_VISIBLE_DEVICES={gpu} sh simclr.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir {out_dir}/pretrain/{dataset} " &
        done
        """
        )
        i += 1
        if i >= len(args.gpus):
            return
        time.sleep(1)

def sweep_alpha(args):
    i = 0
    for dataset in ['cifar10', 'cifar100']:
        # for alpha in '0.05 0.10'.split(' '):
        for alpha in '0.15'.split(' '):
            gpu = args.gpus[i]
            print(rate)
            os.system(f"""
            for file in /data/yfwang/solo-learn/poison_datasets/{dataset}/zoo-simclr/gaussian_noise/{dataset}_zoo-simclr_rate_{rate}_*_{alpha}*.pt
            do 
            # echo ${{file}}
            CUDA_VISIBLE_DEVICES={gpu} sh simclr.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/{dataset} " &
            done
            """
            )
            i += 1
            if i >= len(args.gpus):
                return

def sweep_trigger(args):
    i = 0
    for dataset in ['cifar10', 'cifar100']:
        # for alpha in '0.05 0.10'.split(' '):
        # for alpha in '0.15'.split(' '):
        for trigger in 'checkerboard_1corner checkerboard_4corner checkerboard_center checkerboard_full gaussian_noise'.split(' '):
            gpu = args.gpus[i]
            print(rate)
            os.system(f"""
            for file in {out_dir}/poison_datasets/{dataset}/zoo-simclr/{trigger}/{dataset}_zoo-simclr_rate_{rate}_target_None_trigger_{trigger}_*.pt
            do 
            # rm $file
            # echo ${{file}}
            CUDA_VISIBLE_DEVICES={gpu} sh simclr.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/{dataset} " &
            done
            """
            )
            i += 1
            if i >= len(args.gpus):
                return

def sweep_trial(args):
    i = 0
    # for dataset in ['cifar10', 'cifar100']:
        # for alpha in '0.05 0.10'.split(' '):
        # for alpha in '0.15'.split(' '):
    for budget in '5 50'.split(' '):
        for trial in '0 1 2 3 4'.split(' '):
            gpu = args.gpus[i]
            print(rate)
            os.system(f"""
            for file in {out_dir}/poison_datasets/{dataset}/{poison_method}/gaussian_noise/budget_{budget}/{dataset}_{poison_method}_rate_{rate}_target_0_trigger_gaussian_noise_alpha_0.20_class_0_trial_{trial}_*.pt
            do 
            # rm $file
            # echo ${{file}}
            CUDA_VISIBLE_DEVICES={gpu} sh simclr.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/{dataset} " &
            done
            """
            )
            i += 1
            if i >= len(args.gpus):
                return

if __name__ == "__main__":
    import argparse
    import os 

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('gpus', type=int, nargs="+", help="")

    args = parser.parse_args()
    sweep_pretrain_method(args)
    # sweep_pretrain_method(args)
    # sweep_eval(args)
    # sweep_cifar100(args)
    # sweep_poison_pretrain_method(args)
    # sweep_alpha(args)
    # sweep_poison_rate(args)
    # sweep_trigger(args)
    # sweep_trial(args)