import torch, torch.nn as nn, torch.optim as optim, random, numpy as np, argparse, json, utils.preprocessing as pp, utils.data_helper as dh, os, logging, pandas as pd, pdb
from transformers import AdamW
from utils import modeling, model_eval, utils
from tqdm import tqdm
from pathlib import Path

def run_classifier():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    labels = {
            'PStance'    : ['Joe Biden', 'Bernie Sanders', 'Donald Trump'],
            'AM'         : ['abortion', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms'],
            'SemEval2016': ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion'],
            'Covid19'    : ['face masks', 'fauci', 'stay at home orders', 'school closures'],
            'merged'     : ['Joe Biden', 'Bernie Sanders', 'Donald Trump', 'abortion', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms', 'Atheism', 'Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion', 'face masks', 'fauci', 'stay at home orders', 'school closures'],
            'merged_v2'  : ['Joe Biden', 'Bernie Sanders', 'Donald Trump', 'abortion', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms', 'Atheism', 'Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion', 'face masks', 'fauci', 'stay at home orders', 'school closures'],
            'Stance_Merge': ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion', 'face masks', 'fauci', 'stay at home orders', 'school closures', 'abortion', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms', 'Donald Trump', 'Joe Biden', 'Bernie Sanders'],
            'Stance_Merge_Unrelated': ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'abortion', 'face masks', 'fauci', 'stay at home orders', 'school closures', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms', 'Donald Trump', 'Joe Biden', 'Bernie Sanders', 'Unrelated']
            }

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PStance", help="PStance or SemEval2016 or WTWT or Covid19 or AM or merged")
    parser.add_argument("--test_dataset", type=str, default="", help="PStance or SemEval2016 or WTWT or Covid19 or AM or merged")
    parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_pretrained_path", type=str, default="", help="Path to pretrained model")
    args = parser.parse_args()
    print(args)

    random_seeds     = [args.seed]
    model_select     = args.model_select
    lr               = args.lr
    batch_size       = args.batch_size
    total_epoch      = args.epochs

    # create Normalization Dictionary
    with open("utils/noslang_data.json", "r") as f:
        data1 = json.load(f)
    
    data2 = {}
    with open("utils/emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    
    normalization_dict = {**data1,**data2}

    Path('output/{}/'.format(args.dataset)).mkdir(parents=True, exist_ok=True)
    
    file = 'output/{}/logs_{}.log'.format(args.dataset, args.model_select)
    if os.path.exists(file):    os.remove(file)
    utils.set_logger(file)

    if args.test_dataset == "":
        args.test_dataset = args.dataset

    path      = '../data/{}/'.format(args.dataset)
    test_path = '../data/{}/'.format(args.test_dataset)
    logging.info('Loading dataset {}...'.format(args.dataset))

    best_result, best_val = [], []
    for seed in random_seeds:
        filename1 = path + 'raw_train_all_onecol.csv'
        filename2 = path + 'raw_val_all_onecol.csv'
        filename3 = test_path + 'raw_test_all_onecol.csv'
        
        x_train, x_train_target, x_train_stance = pp.clean_all(filename1, args, normalization_dict)
        x_val,   x_val_target,   x_val_stance   = pp.clean_all(filename2, args, normalization_dict)
        x_test,  x_test_target,  x_test_stance  = pp.clean_all(filename3, args, normalization_dict)

        logging.info('Printing sample data...')
        logging.info("x_train: {}".format(x_train[0]))
        logging.info("x_train_target: {}".format(x_train_target[0]))
        logging.info("x_val: {}".format(x_val[0]))
        logging.info("x_val_target: {}".format(x_val_target[0]))
        logging.info("x_test: {}".format(x_test[0]))
        logging.info("x_test_target: {}".format(x_test_target[0]))

        num_labels  = len(set(x_train_target))
        print('Num labels: ', num_labels)
        x_train_all = [x_train, x_train_target]
        x_val_all   = [x_val,   x_val_target]
        x_test_all  = [x_test,  x_test_target]
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        word_vectors = []
        if args.model_select == 'BiLSTM':
            word_vectors, word_index = dh.build_vocab(x_train, x_val, x_test, labels[args.dataset])
            x_train_all = dh.data_helper_BiLSTM(x_train_all, word_index)
            x_val_all   = dh.data_helper_BiLSTM(x_val_all, word_index)
            x_test_all  = dh.data_helper_BiLSTM(x_test_all, word_index)
            _, y_train, trainloader = dh.data_loader_BiLSTM(x_train_all, batch_size, 'train')
            _, y_val,   valloader   = dh.data_loader_BiLSTM(x_val_all, batch_size, 'val')
            _, y_test,  testloader  = dh.data_loader_BiLSTM(x_test_all, batch_size, 'test')
            model = modeling.BiLSTM_Classifier(num_labels, word_vectors).cuda()

        else:
            x_train_all, x_val_all, x_test_all = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, model_select)
            x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len, y_train, trainloader = dh.data_loader(x_train_all, batch_size, 'train')
            x_val_input_ids,   x_val_seg_ids,   x_val_atten_masks  ,   x_val_len, y_val  , valloader   = dh.data_loader(x_val_all,   batch_size, 'val')
            x_test_input_ids,  x_test_seg_ids,  x_test_atten_masks ,  x_test_len, y_test , testloader  = dh.data_loader(x_test_all,  batch_size, 'test')
            model = modeling.stance_classifier(num_labels,model_select).cuda()

        if model_select == 'BiLSTM':
            optimizer = AdamW(model.parameters(), lr=lr)

        else: # Bert, Bertweet models
            for n,p in model.named_parameters():
                if "bert.embeddings" in n:
                    p.requires_grad = False
            
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}]
            optimizer = AdamW(optimizer_grouped_parameters)
        
        if args.load_pretrained_path != "":
            checkpoint = torch.load(args.load_pretrained_path, map_location=device)
            model.load_state_dict(checkpoint)

        loss_function = nn.CrossEntropyLoss(reduction='sum')
        
        sum_loss, sum_val, train_f1_average, val_f1_average, test_f1_average, confusion_matrices, misclassifications, ground_truths, predictions = [], [], [], [], [[]], [], [], [], []

        for epoch in range(0, total_epoch):
            logging.info('Epoch:{}'.format(epoch))
            
            # training            
            train_loss, valid_loss = [], []
            model.train()
            
            for params in tqdm(trainloader):
                optimizer.zero_grad()
                target  = params[-1]
                params  = params[:-1] # remove target from params and pass rest of the parameters
                output1 = model(*params)
                loss    = loss_function(output1, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_loss.append(loss.item())
            
            sum_loss.append(sum(train_loss)/len(x_train))  
            logging.info(sum_loss[epoch])

            # evaluation on dev set
            model.eval()
            val_preds = []
            
            with torch.no_grad():
                for params in valloader:
                    params = params[:-1]
                    pred1  = model(*params)
                    val_preds.append(pred1)
            
            pred1 = torch.cat(val_preds, 0)
            _, f1_average, _, _, _, _, _, _ = model_eval.compute_f1(pred1,y_val)
            val_f1_average.append(f1_average)
            
            if f1_average == max(val_f1_average):
                print('Best epoch till now: ', epoch)
                # torch.save(model.state_dict(), 'output/models/model_{}_data_{}.pt'.format(args.model_select, args.dataset))

            # evaluation on test set
            with torch.no_grad():
                test_preds = []
                for params in testloader:
                    params = params[:-1]
                    pred1  = model(*params)
                    test_preds.append(pred1)
                pred1 = torch.cat(test_preds, 0)
                pred1_list = [pred1]
                y_test_list = [y_test]

                for ind in range(len(y_test_list)):
                    pred1 = pred1_list[ind]
                    _, f1_average, _, _, confusion_matrix, mislabeled, y_pred, y_true = model_eval.compute_f1(pred1, y_test_list[ind])
                    test_f1_average[ind].append(f1_average)
                    confusion_matrices.append(confusion_matrix)
                    misclassifications.append(mislabeled)
                    ground_truths.append(y_true)
                    predictions.append(y_pred)
        
        best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1] 
        best_result.append([100*f1[best_epoch] for f1 in test_f1_average])

        logging.info("******************************************")
        logging.info("dev results with seed {} on all epochs".format(seed))
        logging.info(val_f1_average)
        best_val.append(val_f1_average[best_epoch])
        logging.info("******************************************")
        logging.info("test results with seed {} on all epochs".format(seed))
        logging.info(test_f1_average)
        
        file = 'output/{}/predictions_{}_seed_{}.csv'.format(args.dataset, args.model_select, args.seed)
        if os.path.exists(file):
            os.remove(file)
        df = pd.DataFrame()
        df['Tweet'] = [' '.join(item) for item in x_test]
        df['Target'] = [labels[args.dataset][j] for j in ground_truths[best_epoch]]
        df['Mapped Target'] = [labels[args.dataset][j] for j in predictions[best_epoch]]
        df['Stance'] = x_test_stance
        df.to_csv(file, index=False)
    
    # model that performs best on the dev set is evaluated on the test set
    logging.info("model performance on the test set: ")
    logging.info('Final Result: {}'.format(best_result))

if __name__ == "__main__":
    run_classifier()
