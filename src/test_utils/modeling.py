import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BartConfig, BartForSequenceClassification, AutoModelForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
from transformers import RobertaModel, AutoModel


# BERTweet or BERT
class bert_classifier(nn.Module):

    def __init__(self, config, plm_model):
        
        super(bert_classifier, self).__init__()
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()
        
        if plm_model == 'bertweet':
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base", local_files_only=True)
        elif plm_model == 'bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        self.bert.pooler = None
        self.linear_main = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out_main = nn.Linear(self.bert.config.hidden_size, int(config['num_labels']))
        self.linear_aux = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out_aux = nn.Linear(self.bert.config.hidden_size, int(config['num_tar']))
        
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len, task_id):

        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls = last_hidden[0][:,0]
        query = self.dropout(cls)
        
        if task_id[0] == 0:
            linear = self.relu(self.linear_main(query))
            out = self.out_main(linear)
        else:
            linear = self.relu(self.linear_aux(query))
            out = self.out_aux(linear)
        
        return out

    
# BiLSTM, TAN, BiCond and CrossNet
class lstm_classifier(nn.Module):

    def __init__(self, config, model):

        super(lstm_classifier, self).__init__()
        self.embedding = nn.Embedding(int(config['num_vocab']), 300)
        self.model = model
        self.hidden_size = int(config['hidden_size'])
        self.linear_size = int(config['linear_size'])
        self.dropout = nn.Dropout(float(config['dropout']))
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(300, self.hidden_size, bidirectional=True)
        self.linear_main = nn.Linear(self.hidden_size*2, self.linear_size)
        self.out_main = nn.Linear(self.linear_size, int(config['num_labels']))
        self.linear_aux = nn.Linear(self.hidden_size*2, self.linear_size)
        self.out_aux = nn.Linear(self.linear_size, int(config['num_tar']))
        
        if model == 'tan':
            self.atten_tan = nn.Linear(600,1)
        if model == 'bice':
            self.lstm2 = nn.LSTM(300, self.hidden_size, bidirectional=True)
        elif model == 'crossnet':
            self.atten_crossnet_main = nn.Linear(self.hidden_size*2, 1) 
            self.atten_crossnet_aux = nn.Linear(self.hidden_size*2, 1) 
            self.linear_atten = nn.Linear(self.hidden_size*2, self.hidden_size*2) 
            self.linear_atten2 = nn.Linear(self.hidden_size*2, self.hidden_size*2) 
            self.lstm = nn.LSTM(300, self.hidden_size, bidirectional=True, batch_first=True)
            self.lstm2 = nn.LSTM(300, self.hidden_size, bidirectional=True, batch_first=True)
        
    def forward(self, x, target_word, x_len, task_id):
        
        b_size = x.shape[0]
        x_embedding = self.embedding(x)  
        x_embedding = self.dropout(x_embedding) 
        
        if self.model == 'bilstm':
            if task_id[0] == 0:
                lstm_out, _ = self.lstm(x_embedding.transpose(0,1))
                lstm_out = lstm_out.transpose(0,1)  
                cat = torch.cat((lstm_out[:,-1,:self.hidden_size], lstm_out[:,0,self.hidden_size:]), 1)
                linear = self.relu(self.linear_main(cat))
                out = self.out_main(linear)
            else:
                lstm_out, _ = self.lstm(x_embedding.transpose(0,1))
                lstm_out = lstm_out.transpose(0,1)  
                cat = torch.cat((lstm_out[:,-1,:self.hidden_size], lstm_out[:,0,self.hidden_size:]), 1)          
                linear = self.relu(self.linear_aux(cat))
                out = self.out_aux(linear)
                
        elif self.model == 'tan':
            if task_id[0] == 0:
                t_embedding = self.embedding(target_word)
                t_embedding = torch.mean(t_embedding, dim=1, keepdim=True)
                xt_embedding = torch.cat((x_embedding, t_embedding.expand(b_size,x.size()[1],-1)), dim=2)
                lstm_out, _ = self.lstm(x_embedding.transpose(0,1))
                lstm_out = lstm_out.transpose(0,1)  
                atten = self.atten_tan(xt_embedding).squeeze(2)
                final_hidden_state = torch.bmm(F.softmax(atten, dim=1).unsqueeze(1), lstm_out).squeeze(1)
                linear = self.relu(self.linear_main(final_hidden_state))
                out = self.out_main(linear)
            else:
                lstm_out, _ = self.lstm(x_embedding.transpose(0,1))
                lstm_out = lstm_out.transpose(0,1)  
                cat = torch.cat((lstm_out[:,-1,:self.hidden_size], lstm_out[:,0,self.hidden_size:]), 1)
                linear = self.relu(self.linear_aux(cat))
                out = self.out_aux(linear)
        
        elif self.model == 'bice':
            if task_id[0] == 0:
                t_embedding = self.embedding(target_word)
                _, (h_n, c_n) = self.lstm(t_embedding.transpose(0,1))
                lstm_out, _ = self.lstm2(x_embedding.transpose(0,1), (h_n, c_n))
                cat = torch.cat((lstm_out[-1,:,:self.hidden_size], lstm_out[0,:,self.hidden_size:]), 1)
                linear = self.relu(self.linear_main(cat))
                out = self.out_main(linear)
            else:
                lstm_out, _ = self.lstm2(x_embedding.transpose(0,1))
                cat = torch.cat((lstm_out[-1,:,:self.hidden_size], lstm_out[0,:,self.hidden_size:]), 1)
                linear = self.relu(self.linear_aux(cat))
                out = self.out_aux(linear)
        
        elif self.model == 'crossnet':
            if task_id[0] == 0:
                t_embedding = self.embedding(target_word)
                _, (h_n, c_n) = self.lstm(t_embedding)
                lstm_out, _ = self.lstm2(x_embedding, (h_n, c_n))
                atten = F.softmax(self.atten_crossnet_main(self.relu(self.linear_atten(lstm_out))).squeeze(2))
                context_vec = torch.einsum('blh,bl->bh', lstm_out, atten)
                linear = self.relu(self.linear_main(context_vec))
                out = self.out_main(linear)
            else:
                lstm_out, _ = self.lstm2(x_embedding)
                atten = F.softmax(self.atten_crossnet_aux(self.relu(self.linear_atten2(lstm_out))).squeeze(2))
                context_vec = torch.einsum('blh,bl->bh', lstm_out, atten)
                linear = self.relu(self.linear_aux(context_vec))
                out = self.out_aux(linear)
              
        return out