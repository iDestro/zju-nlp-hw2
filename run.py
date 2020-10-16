# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from TextCNN import Config, Model
from utils import get_time_dif, Preprocess
import argparse

# 使用python的argparse使程序更具可读性和易操作性
parser = argparse.ArgumentParser(description='Chinese Text Classification')
# 选择需要训练和测试的数据集（英文：en和中文：zh）
parser.add_argument('--language', type=str, required=True, help='choose zh or en language~')
args = parser.parse_args()

if __name__ == '__main__':

    # 根据命令行的参数选择需要训练或测试的数据集
    dataset = './aclImdb' if args.language == 'en' else './THUCNews'
    # 固定随机种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 根据数据集的不同构建相应的配置类
    config = Config(dataset)
    # 记录加载数据开始时间
    start_time = time.time()
    print("Loading data...")
    # 根据配置类构建预处理类
    preprocess = Preprocess(config)
    # 该预处理类中的一个成员方法可以获取训练、验证和测试的Dataloader
    train_iter, dev_iter, test_iter = preprocess.get_iters()
    # 记录加载数据的结束时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 根据配置类新建TextCNN模型
    model = Model(config).to(config.device)
    # 初始化网络参数
    init_network(model)
    print(model.parameters)
    # 开始训练
    train(config, model, train_iter, dev_iter, test_iter)
