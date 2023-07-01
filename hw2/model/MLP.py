import torch.nn as nn
import torch
import torch.nn.functional as F
from model.config import CONFIG
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, config:CONFIG):
        super(MLP, self).__init__()
        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        hidden_dims = config.hidden_dims
        n_class = config.n_class
        drop_keep_prob = config.drop_keep_prob

        self.__name__ = 'MLP'
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.relu = nn.ReLU()
        self.first = nn.Linear(embedding_dim,hidden_dims[0])
        self.mlp_layer =  nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        )
        self.dropout = nn.Dropout(drop_keep_prob)
        self.last = nn.Linear(hidden_dims[-1], n_class)
        #init weights
        for _, p in self.named_parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, mean = 0, std = 0.01)

    def forward(self, inputs):
        output = self.first(self.embedding(inputs.to(torch.int64)))
        for _, mlp in enumerate(self.mlp_layer):
            output = self.relu(output)
            output = self.dropout(output)
            output = mlp(output)
        output = output.permute(0, 2, 1)    # batch * h * len
        return self.last(F.max_pool1d(output, output.shape[2]).squeeze(2))
    
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            if self.config.init_weight == "kaiming":
                init.kaiming_uniform_(m.weight)
            else:
                init.xavier_uniform_(m.weight)
            # or xavier_uniform_
            init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)