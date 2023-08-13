import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import preprocessor as p
import random
import csv
import os
import ast
import argparse
import numpy as np
import pandas as pd
import warnings
import train_utils.preprocessing as pp
import train_utils.data_helper as dh
from train_utils import modeling, metrics, model_utils

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def train():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-m', '--model_select', help='Model name', required=False)
    parser.add_argument('-mod_dir', '--model_dir', help='Saved model dir', required=False)
    parser.add_argument('-train', '--train_data', help='Name of the train data file', default=None, required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    parser.add_argument('-a', '--aux_eval', help='Auxiliary task', action='store_true')
    parser.add_argument('-mul', '--mul_task', help='Multi-task with target prediction as aux task', action='store_true')
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
    mul_task = args['mul_task']
    batch_size = int(config['batch_size'])
    print("Model: ",model_select)
    print("Batch size: ",config['batch_size'])
    print("Multi-task learning: ",args['mul_task'])
    print(60*"#")
    
    # load train/val/test sets
    file = [args['train_data'], args['dev_data'], args['test_data']]
    print(outdir, file[0], file[1], file[2])
    if model_select.startswith('bert'):
        x_train_all, x_val_all, x_test_all, x_train_aux_all, _ = dh.load_dataset(file, model_select, config)
    else:
        x_train_all, x_val_all, x_test_all, x_train_aux_all, _, word_vectors = dh.load_dataset(file, model_select, config)
    split_point = len(x_train_all[0])
    
    if mul_task:
        x_train_all = [a + b for a, b in zip(x_train_all,x_train_aux_all)]
        
    if model_select.startswith('bert'):
        _, _, _, y_train, _, trainloader = dh.data_loader(x_train_all, batch_size, 'train', model_select, mul_task, split_point)
        _, _, _, y_val, _, valloader = dh.data_loader(x_val_all, batch_size, 'val', model_select)   
        _, _, _, y_test, _, testloader = dh.data_loader(x_test_all, batch_size, 'test', model_select)   
    else:
        _, y_train, _, _, trainloader = dh.data_loader(x_train_all, batch_size, 'train', model_select, mul_task, split_point)
        _, y_val, _, _, valloader = dh.data_loader(x_val_all, batch_size, 'val', model_select) 
        _, y_test, _, _, testloader = dh.data_loader(x_test_all, batch_size, 'test', model_select) 
    y_val = y_val.to(device)
    y_test = y_test.to(device)

    # test
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
        model, optimizer = model_utils.model_setup(config, model_select, device)
        if model_select not in ['bert','bertweet']:
            et = torch.tensor(list(word_vectors.values()), dtype=torch.float32).cuda()
            model.embedding.weight = nn.Parameter(et, requires_grad = False)
        loss_function = nn.CrossEntropyLoss()
        kwargs = {
                    "model": model,
                    "optimizer": optimizer,
                    "model_select": model_select,
        }
        updater = model_utils.model_updater(**kwargs)
        
        best_val = 0
        best_test_micro, best_test_macro = [], []
        for epoch in range(0, int(config['total_epochs'])):
            print('Epoch:', epoch)

            # train
            updater.model.train()
            sum_loss = updater.update(trainloader, loss_function, device)
            print(sum_loss/len(y_train))

            # evaluation on validation set
            updater.model.eval()
            with torch.no_grad():
                preds = model_utils.model_preds(valloader, updater.model, device, model_select)
                f1_average = metrics.train_compute_f1(preds, y_val)
                
            if f1_average > best_val:
                best_val = f1_average
                model_weight = os.path.join(outdir,model_select+'_seed{}.pt'.format(seed))
                torch.save(updater.model.state_dict(), model_weight)

        print("Best val results of model {} and seed {} are: {}".format(model_select, seed, best_val))
        
        # evaluation on test set 
        weight = os.path.join(outdir, model_select+'_seed{}.pt'.format(seed))
        model.load_state_dict(torch.load(weight))

        model.eval()
        with torch.no_grad():
            preds = model_utils.model_preds(testloader, model, device, model_select)

            # micro-averaged F1
            f1_average = metrics.train_compute_f1(preds, y_test)
            best_test_micro.append(f1_average)

            # macro-averaged F1
            preds_list = dh.sep_test_set(preds) 
            y_test_list = dh.sep_test_set(y_test)
            temp_list = []
            for ind in range(len(y_test_list)):
                f1_average = metrics.train_compute_f1(preds_list[ind], y_test_list[ind])
                temp_list.append(f1_average)
            best_test_macro.append(sum(temp_list)/len(temp_list))
                
        print("Best micro test results: " + ",".join(map(str, best_test_micro)))
        print("Best micro test results on SemEval-2016: " + ",".join(map(str, [temp_list[0]])))
        print("Best micro test results on COVID-19: " + ",".join(map(str, [temp_list[1]])))
        print("Best micro test results on argmin: " + ",".join(map(str, [temp_list[2]])))
        print("Best micro test results on PStance: " + ",".join(map(str, [temp_list[3]])))
        print("Best macro test results: " + ",".join(map(str, best_test_macro)))
    
if __name__ == "__main__":
    train()