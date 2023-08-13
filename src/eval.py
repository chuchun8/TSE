import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import preprocessor as p
import random
import csv
import os
import argparse
import numpy as np
import pandas as pd
import warnings
import test_utils.preprocessing as pp
import test_utils.data_helper as dh
from test_utils import modeling, metrics, model_utils

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def evaluation():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-m', '--model_select', help='Model name', required=False)
    parser.add_argument('-mod_dir', '--model_dir', help='Saved model dir', required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    parser.add_argument('-a', '--aux_eval', help='Auxiliary task', action='store_true')
    args = vars(parser.parse_args())

    # gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    
    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    # print parameters in the log file
    random_seeds = []
    random_seeds.append(int(args['seed']))
    outdir = args['model_dir']
    model_select = args['model_select']
    batch_size = int(config['batch_size'])
    print("Model: ",model_select)
        
    # load test set
    if model_select.startswith('bert'):
        x_test_all, gt_tar, map_tar = dh.load_dataset(args['test_data'], model_select, config)
        _, _, _, y_test, _, testloader = dh.data_loader(x_test_all, batch_size, 'test', model_select)
    else:
        x_test_all, word_vectors, gt_tar, map_tar = dh.load_dataset(args['test_data'], model_select, config)
        _, y_test, _, _, testloader = dh.data_loader(x_test_all, batch_size, 'test', model_select)     
    y_test = y_test.to(device) 

    # test
    best_micro_result = []
    for seed in random_seeds:    
        print("current random seed: ", seed)
        
        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # model setup
        weight = os.path.join(outdir,model_select+'_seed{}.pt'.format(seed))
        if model_select in ['bert','bertweet']:
            model = modeling.bert_classifier(config, model_select).to(device)
        else:
            model = modeling.lstm_classifier(config, model_select).to(device)
        model.load_state_dict(torch.load(weight))
        if model_select not in ['bert','bertweet']:
            et = torch.tensor(list(word_vectors.values()), dtype=torch.float32).cuda()
            model.embedding.weight = nn.Parameter(et, requires_grad = False)

        # evaluation
        model.eval()
        with torch.no_grad():
            outputs = model_utils.model_preds(testloader, model, device, model_select)
            print(outputs.size(),y_test.size())
            f1_average = metrics.eval_compute_f1(outputs, y_test, gt_tar, map_tar)
            
        best_micro_result.append(f1_average)
    
        print("Final results on {} of model {} and seed {} are: {}".format(args['test_data'], model_select, seed, best_micro_result))
    
if __name__ == "__main__":
    evaluation()