import torch
import transformers
import json
import random
import numpy as np
import gensim.models.keyedvectors as word2vec
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Sampler
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer, BartTokenizer,RobertaTokenizer
from torchtext.legacy import data
from train_utils import preprocessing as pp
transformers.logging.set_verbosity_error()


def convert_data_to_ids(tokenizer, text, task):
    
    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []  
    if task == 'main':
        # Main task stance detection
        for tar, sent in zip(text[2], text[0]):
            encoded_dict = tokenizer.encode_plus(
                                ' '.join(tar),
                                ' '.join(sent),             # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 128,           # Pad & truncate all sentences.
                                padding = 'max_length',
                                return_attention_mask = True,   # Construct attn. masks.
                                truncation = True,
                           )

            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            seg_ids.append(encoded_dict['token_type_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            sent_len.append(sum(encoded_dict['attention_mask']))
    else:
        # Auxiliary task target prediction
        for sent in text[0]:
            encoded_dict = tokenizer.encode_plus(
                                ' '.join(sent),             # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 128,           # Pad & truncate all sentences.
                                padding = 'max_length',
                                return_attention_mask = True,   # Construct attn. masks.
                                truncation = True,
                           )

            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            seg_ids.append(encoded_dict['token_type_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            sent_len.append(sum(encoded_dict['attention_mask']))

    return input_ids, seg_ids, attention_masks, sent_len


def build_vocab(x_train, x_val, x_test, x_train_target, x_train2):    
    
    # Build vocabulary for baselines
    model1 = word2vec.KeyedVectors.load_word2vec_format('./crawl-300d-2M.bin', limit = 500000, binary=True)
    text_field = data.Field(lower=True)
    text_field.build_vocab(x_train, x_val, x_test, x_train_target, ['<pad>'], x_train2)
    word_vectors, word_index = dict(), dict()
    ind = 0
    for word in text_field.vocab.itos:
        if word in model1.vocab:
            word_vectors[word] = model1[word]
        elif word == '<pad>':
            word_vectors[word] = np.zeros(300, dtype=np.float32)
        else:
            word_vectors[word] = np.random.uniform(-0.25, 0.25, 300)
        word_index[word] = ind
        ind = ind + 1
    
    return word_vectors, word_index


# Prepare data for BERT/BERTweet
def data_helper_bert(x_all, plm_model, task='main'):
    
    if plm_model == 'bertweet':
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, local_files_only=True)
    elif plm_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, local_files_only=True)
    
    print("Length of the set: %d"%(len(x_all[0])))
    x_input_ids, x_seg_ids, x_atten_masks, x_len = convert_data_to_ids(tokenizer, x_all, task)
    x_all = [x_input_ids, x_seg_ids, x_atten_masks, x_all[1], x_len]
    
    return x_all


# Prepare data for the rest baselines
def data_helper(config, x_all, word_index, task='main'):
    
    if task == 'aux':
        x, y = x_all[0], x_all[1]
        x_target = [['<pad>']] * len(y)
    else:
        x, y, x_target = x_all[0], x_all[1], x_all[2]
    print("Length of the set: %d"%(len(x)))
    sequence_length = int(config['sent_len'])
    
    # Get sequence length for each sentence
    x_len = [len(xi) if len(xi) <= sequence_length else sequence_length for xi in x]

    # Padding
    x = [xi[:sequence_length] for xi in x]
    x_pad = [xi[:sequence_length] + ['<pad>'] * (sequence_length - len(xi)) for xi in x]
    x_target_pad = [xi[:5] + ['<pad>'] * (5 - len(xi)) for xi in x_target]
    
    # Convert word to index
    x_index = [[word_index[word] for word in sentence] for sentence in x_pad]
    x_target_index = [[word_index[word] for word in sentence] for sentence in x_target_pad]
    
    x_data_all = [x_index, y, x_target_index, x_len]
    
    return x_data_all


def chunk(indices, chunk_size):
    
    return torch.split(torch.tensor(indices), chunk_size)


# Task sampler for multi-task setting
class MultiTaskSampler(Sampler):
    
    def __init__(self, dataset, batch_size, halfway_point):
        self.first_half_indices = list(range(halfway_point)) # Samples of main task
        self.second_half_indices = list(range(halfway_point, len(dataset))) # Samples of auxiliary task
        self.batch_size = batch_size
        
    def __iter__(self):
        random.shuffle(self.first_half_indices)
        random.shuffle(self.second_half_indices)
        first_half_batches  = chunk(self.first_half_indices, self.batch_size)
        second_half_batches = chunk(self.second_half_indices, self.batch_size)
        combined = list(first_half_batches + second_half_batches)
        combined = [batch.tolist() for batch in combined]
        random.shuffle(combined)
        
        return iter(combined)
    
    def __len__(self):
        
        return (len(self.first_half_indices) + len(self.second_half_indices)) // self.batch_size

    
def data_loader(x_all, batch_size, mode, plm_model, mul_task=False, split_point=0):
    
    if plm_model.startswith('bert'):
        x_input_ids = torch.tensor(x_all[0], dtype=torch.long)
        x_seg_ids = torch.tensor(x_all[1], dtype=torch.long)
        x_atten_masks = torch.tensor(x_all[2], dtype=torch.long)
        y = torch.tensor(x_all[3], dtype=torch.long)
        x_len = torch.tensor(x_all[4], dtype=torch.long)
    else:
        x = torch.tensor(x_all[0], dtype=torch.long)
        y = torch.tensor(x_all[1], dtype=torch.long)
        x_target = torch.tensor(x_all[2], dtype=torch.long)
        x_len = torch.tensor(x_all[3], dtype=torch.long)
    
    if not mul_task:
        task_id = torch.tensor([0] * len(y), dtype=torch.long)
        if plm_model.startswith('bert'): 
            tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, x_len, task_id)
        else:
            tensor_loader = TensorDataset(x, y, x_target, x_len, task_id)
            
        if mode == 'train':
            loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
        else:
            loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
    else:
        task_id = torch.tensor([0] * split_point + [1] * (len(y)-split_point), dtype=torch.long)
        if plm_model.startswith('bert'): 
            tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, x_len, task_id)
        else:
            tensor_loader = TensorDataset(x, y, x_target, x_len, task_id)
            
        multi_task_batch_sampler = MultiTaskSampler(tensor_loader, batch_size, split_point)
        loader = DataLoader(tensor_loader, batch_sampler=multi_task_batch_sampler)
    
    if plm_model.startswith('bert'):
        
        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, loader

    else:
        
        return x, y, x_target, x_len, loader


# Evaluation on SemEval-2016, COVID-19, AM and P-Stance, respectively
def sep_test_set(input_data):
    
    return [input_data[:1080], input_data[1080:1880], input_data[1880:6989], input_data[6989:9146]]


def load_dataset(filename, plm_model, config):

    # Create normalization dictionary for preprocessing
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}
    
    # Load train/val/test sets
    x_train, y_train, x_train_target, x_train_aux, y_train_aux = pp.clean_all(filename[0], normalization_dict)
    x_val, y_val, x_val_target, _, _ = pp.clean_all(filename[1], normalization_dict)
    x_test, y_test, x_test_target, _, y_test_aux = pp.clean_all(filename[2], normalization_dict)
    x_train_all = [x_train, y_train, x_train_target]
    x_val_all = [x_val, y_val, x_val_target]
    x_test_all = [x_test, y_test, x_test_target]
    x_train_aux_all = [x_train_aux, y_train_aux] # auxiliary target prediction task

    if plm_model.startswith('bert'):
        x_train_all = data_helper_bert(x_train_all, plm_model, 'main')
        x_val_all = data_helper_bert(x_val_all, plm_model, 'main')
        x_test_all = data_helper_bert(x_test_all, plm_model, 'main')
        x_train_aux_all = data_helper_bert(x_train_aux_all, plm_model, 'aux')
        
        return x_train_all, x_val_all, x_test_all, x_train_aux_all, y_test_aux
    
    else:
        word_vectors, word_index = build_vocab(x_train, x_val, x_test, x_train_target, x_train_aux)
        x_train_all = data_helper(config, x_train_all, word_index, 'main')
        x_val_all = data_helper(config, x_val_all, word_index, 'main')
        x_test_all = data_helper(config, x_test_all, word_index, 'main')
        x_train_aux_all = data_helper(config, x_train_aux_all, word_index, 'aux')
        
        return x_train_all, x_val_all, x_test_all, x_train_aux_all, y_test_aux, word_vectors
