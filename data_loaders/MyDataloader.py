import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random

from utils.general import parse_yaml
from data_sets.dataset import Select_Dataset

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders  获取环境变量 不存在返回默认值"True"


# 未考虑分布式 多卡
# 未考虑数据类别平衡
# dataloador workers 未设置


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def create_dataloader(data,  # yaml文件
                      trainorval, # "train" or "test" or "val" 与yaml对应
                      batch_size,
                      # hyp=hyp, #是否数据处理
                      augment=False,
                      workers=8,
                      shuffle=True,
                      seed=0,
                      LOGGER=None):

    data_yaml = parse_yaml(data)
    data_rootdir = data_yaml['data_path']
    data_type = data_yaml['data_type']   # data_type = "Indoor_5G_scaters"
    data_trainorval = data_yaml[trainorval]  # 训练测试数据路径   处理数据交给 Dataset 类

    # 需要import 对应的Dataset类；Dataset(root_dir)
    dataset = Select_Dataset(data_type)()(os.path.join(data_rootdir, data_trainorval))  # Dataset类
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None # 不考虑分布式
    loader = DataLoader # 不考虑数据类别平衡
    generator = torch.Generator() # 数据集加载方式 生成随机数 索引  # shuffle 每个epoch开始时对数据集进行随机打乱
    generator.manual_seed(6148914691236517205 + seed) # 固定seed 可复现

    LOGGER.info(f'{data_type} {trainorval} dataset with {len(dataset)} samples are loaded, batchsize is {batch_size}.')

    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY, # 是否放在固定的内存位置加速GPU访问，会占用内存，数据小True，大False
                  collate_fn=None, # 是否进行自定义样本批处理 一个batch的处理方式
                  worker_init_fn=seed_worker, #  DataLoader 的工作进程（worker）初始化时执行的函数
                  generator=generator), dataset