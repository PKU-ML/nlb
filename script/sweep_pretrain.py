import os

for i in range(6):
    os.system(f"nohup python3 script/sweep_pretrain_single.py {i} &")