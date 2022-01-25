from torch import nn
import torch.nn.functional as F
import torch

# this implementation separates scopeit model from the embedder (bert).
# We are doing this to avoid using torch.nn.dataparallel since it has some problems.
# see: https://github.com/pytorch/pytorch/issues/7092#issuecomment-385357970
# this problem isn't solved when flatten_parameters is used too.
class ScopeIt(nn.Module):
    def __init__(self, bert_hidden_size, hidden_size, num_layers=1, dropout=0.1, num_token_labels=15):
        super(ScopeIt, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = bert_hidden_size
        self.bigru1 = nn.GRU(self.embedding_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.bigru2 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

        # In case we use the biGRU's output for token classification
        self.token_boomer = MLP(hidden_size * 2, dim_feedforward=hidden_size*2*4, dropout=dropout, shortcut=False)
        self.token_linear = nn.Linear(self.hidden_size * 2, num_token_labels)

        # In case we use BERT embeddings for token classification
        # self.token_boomer = MLP(hidden_size, dim_feedforward=hidden_size*4, dropout=dropout, shortcut=True)
        # self.token_linear = nn.Linear(self.hidden_size, num_token_labels)

        self.sent_boomer = MLP(hidden_size * 2, dim_feedforward=hidden_size*2*4, dropout=dropout, shortcut=False)
        self.sent_linear = nn.Linear(self.hidden_size * 2, 1)

        self.doc_boomer = MLP(hidden_size, dim_feedforward=hidden_size*4, dropout=dropout, shortcut=False)
        self.doc_linear = nn.Linear(self.hidden_size, 1)

    def forward(self, embeddings): # embeddings ->[Sentences, SeqLen, BERT_Hidden]

        # In case we use the biGRU's output for token classification
        bigru1_all_hiddens, bigru1_last_hidden = self.bigru1(embeddings) # pass the output of bert through the first bigru to get sentence and token embeddings
        # bigru1_all_hiddens -> [Sentences, SeqLen, Hidden * 2]
        boomed_tokens = self.token_boomer(bigru1_all_hiddens) # boomed_tokens -> [Sentences, SeqLen, Hidden * 2]
        token_logits = self.token_linear(boomed_tokens) # token_logits -> [Sentences, SeqLen, num_token_labels]

        # In case we use BERT embeddings for token classification
        # bigru1_last_hidden = self.bigru1(embeddings)[1] # pass the output of bert through the first bigru to get sentence embeddings
        # boomed_tokens = self.token_boomer(embeddings)
        # token_logits = self.token_linear(boomed_tokens)

        sent_embeddings = bigru1_last_hidden[0, :, :] + bigru1_last_hidden[1, :, :] # here we add the output of two GRUs (forward and backward)
        # sent_embeddings -> [Sentences, HiddenSize]

        bigru2_output = self.bigru2(sent_embeddings.unsqueeze(0))
        boomed_sents = self.sent_boomer(bigru2_output[0])# [Sentences, HiddenSize]

        doc_embeddings = bigru2_output[1][0, :, :] + bigru2_output[1][1, :, :] # here we add the output of two GRUs (forward and backward)
        boomed_doc = self.doc_boomer(doc_embeddings)
        # boomed_doc -> [1, HiddenSize]

        sent_logits = self.sent_linear(boomed_sents).squeeze(0)# [Sentences, 2]
        doc_logit = self.doc_linear(boomed_doc)

        return token_logits, sent_logits, doc_logit # multi-task model training
        # return token_logits, sent_logits, doc_logit, sent_embeddings # coref_head_training aftergru1
        # return token_logits, sent_logits, doc_logit, bigru2_output[0].squeeze(0) # coref_head_training aftergru2


class MLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()
        #self.act = nn.Tanh()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        z = self.linear2(x)

        return z
