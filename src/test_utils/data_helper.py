import torch
import transformers
import json
import numpy as np
import gensim.models.keyedvectors as word2vec
from test_utils import preprocessing as pp
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer, BartTokenizer,RobertaTokenizer
from torchtext.legacy import data
transformers.logging.set_verbosity_error()


def convert_data_to_ids(tokenizer, text):
    
    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []  
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

    return input_ids, seg_ids, attention_masks, sent_len


def build_vocab(x_test, x_test_target, x_test_mapped_tar):    
    
    # Build vocabulary
    model1 = word2vec.KeyedVectors.load_word2vec_format('./crawl-300d-2M.bin', limit = 500000, binary=True)
    text_field = data.Field(lower=True)
    text_field.build_vocab(x_test, x_test_target, x_test_mapped_tar, ['<pad>'])
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
def data_helper_bert(x_all, plm_model):
    
    if plm_model == 'bertweet':
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, local_files_only=True)
    elif plm_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, local_files_only=True)
    
    print("Length of the set: %d"%(len(x_all[0])))
    x_input_ids, x_seg_ids, x_atten_masks, x_len = convert_data_to_ids(tokenizer, x_all)
    x_all = [x_input_ids, x_seg_ids, x_atten_masks, x_all[1], x_len]
    
    return x_all


# Prepare data for the rest baselines
def data_helper(config, x_all, word_index):
    
    print('Loading data')    
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


def data_loader(x_all, batch_size, mode, plm_model):
    
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
    
    task_id = torch.tensor([0] * len(y), dtype=torch.long)
    if plm_model.startswith('bert'): 
        tensor_loader = TensorDataset(x_input_ids, x_seg_ids, x_atten_masks, y, x_len, task_id)
    else:
        tensor_loader = TensorDataset(x, y, x_target, x_len, task_id)

    if mode == 'train':
        loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
    
    if plm_model.startswith('bert'):
        
        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, loader
    
    else:
        
        return x, y, x_target, x_len, loader

    
def load_dataset(filename, plm_model, config):

    # Creating Normalization Dictionary
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}
    
    # Load test set
    x_test, y_test, x_test_target, x_test_mapped_tar = pp.clean_all(filename, normalization_dict)
    x_test_all = [x_test, y_test, x_test_mapped_tar]
    
    if plm_model.startswith('bert'):
        x_test_ind_all = data_helper_bert(x_test_all, plm_model)
        
        return x_test_ind_all, x_test_target, x_test_mapped_tar
    
    else:
        word_vectors, word_index = build_vocab(x_test, x_test_target, x_test_mapped_tar)
        x_test_ind_all = data_helper(config, x_test_all, word_index)
        
        return x_test_ind_all, word_vectors, x_test_target, x_test_mapped_tar