rate = '0.60'
dataset = 'cifar10'
# poison_method = 'zoo-simclr'
poison_method = 'clb'
import time

def sweep_poison_rate(args):
    i = 0
    # for dataset in ['cifar10', 'cifar100']:
    for dataset in ['cifar100']:
        for rate in '0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'.split(' '):
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
    for method in ['sup', 'supcon', 'simclr', 'mocov2plus', 'byol', 'simsiam', 'swav', 'dino', 'barlow']:
        # for rate in '0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'.split(' '):
        gpu = args.gpus[i]
        print(rate)

        os.system(f"""
        for file in /data/yfwang/solo-learn/poison_datasets/{dataset}/{poison_method}/gaussian_noise/{dataset}_{poison_method}_rate_{rate}_*.pt
        do 
        # echo ${{file}}, {method}
        # CUDA_VISIBLE_DEVICES={gpu} sh {method}.sh {dataset} " --poison_data ${{file}}  --use_poison --checkpoint_dir /data/yfwang/solo-learn/pretrain/{dataset} " &
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
                    for file in /data/yfwang/solo-learn/poison_datasets/{dataset}/{poison_method}/gaussian_noise/{dataset}_{poison_method}_rate_{rate}_*.pt
                    do 
                    # echo ${{file}}, {method}
                    CUDA_VISIBLE_DEVICES={gpu} sh {method}.sh {dataset} " --poison_data ${{file}}  --{apply_method} --checkpoint_dir /data/yfwang/solo-learn/pretrain/{dataset} " &
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
                    for file in /data/yfwang/solo-learn/poison_datasets/{dataset}/{poison_method}/gaussian_noise/{dataset}_{poison_method}_rate_{rate}_*.pt
                    do 
                    # echo ${{file}}, {method}
                    CUDA_VISIBLE_DEVICES={gpu} sh {method}.sh {dataset} " --poison_data ${{file}}  --{apply_method} --checkpoint_dir /data/yfwang/solo-learn/pretrain/{dataset} " &
                    done
                    """
                    )
                    i += 1
                    if i >= len(args.gpus):
                        return
                    time.sleep(1)

if __name__ == "__main__":
    import argparse
    import os 

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('gpus', type=int, nargs="+", help="")

    args = parser.parse_args()
    # sweep_pretrain_method(args)
    # sweep_pretrain_method(args)
    # sweep_eval(args)    
    sweep_cifar100(args)