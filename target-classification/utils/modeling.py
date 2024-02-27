import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, BertModel

# BERT/BERTweet
class stance_classifier(nn.Module):

    def __init__(self,num_labels,model_select):

        super(stance_classifier, self).__init__()
        
        self.dropout = nn.Dropout(0.)
        self.relu    = nn.ReLU()
        self.tanh    = nn.Tanh()
        
        if model_select == 'Bertweet':
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        elif model_select == 'Bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out    = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len):
        
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        query       = last_hidden[0][:,0]
        query       = self.dropout(query)
        linear      = self.relu(self.linear(query))
        out         = self.out(linear)
        return out


class BiLSTM_Classifier(nn.Module):

    def __init__(self, num_labels, word_vectors):

        super(BiLSTM_Classifier, self).__init__()
        
        self.hidden_size = 300
        self.linear_size = 300
        self.embedding_dim = 300
        self.embedding     = nn.Embedding(len(list(word_vectors)), self.embedding_dim)
        self.lstm          = nn.LSTM(self.embedding_dim, self.hidden_size, dropout=0., bidirectional=True, batch_first=True)
        self.linear        = nn.Linear(self.hidden_size*2, self.linear_size)
        self.out           = nn.Linear(self.linear_size, num_labels)
        self.relu          = nn.ReLU()
        self.dropout       = nn.Dropout()

    def forward(self, x):

        x    = self.embedding(x)
        x, (hn, cn) = self.lstm(x.float())
        x = hn.transpose(0, 1).reshape(x.shape[0], -1)
        x    = self.dropout(self.relu(self.linear(x)))
        out  = self.out(x)

        return out