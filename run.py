# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from TextCNN import Config, Model
from utils import get_time_dif, Preprocess
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--language', type=str, required=True, help='choose zh or en language~')
args = parser.parse_args()

if __name__ == '__main__':

    dataset = './aclImdb' if args.language == 'en' else './THUCNews'
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = Config(dataset)
    start_time = time.time()
    print("Loading data...")
    preprocess = Preprocess(config)
    train_iter, dev_iter, test_iter = preprocess.get_iters()
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
