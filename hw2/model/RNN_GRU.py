import torch.nn as nn
import torch
from model.config import CONFIG
import torch.nn.init as init


class RNN_GRU(nn.Module):
    def __init__(self, config:CONFIG):

        super(RNN_GRU, self).__init__()

        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.__name__ = 'RNN_GRU'

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        # (seq_len, batch, embed_dim) -> (seq_len, batch, 2 * hidden_size)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc = nn.Linear(64, self.n_class)

    def forward(self, inputs):
        x = self.embedding(inputs.to(torch.int64)).permute(1, 0, 2) 
        h_0 = torch.rand(self.num_layers * 2, x.size(1), self.hidden_size).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        _, h_n = self.encoder(x, h_0)          
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)    
        return (self.fc(self.decoder(torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1))))

    def weight_init(m,self):
        if isinstance(m, nn.Linear):
            if self.config.init_weight == "kaiming":
                init.kaiming_uniform_(m.weight)
            else:
                init.xavier_uniform_(m.weight)
            # or xavier_uniform_
            init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            init.orthogonal_(m.weight_ih_l0)
            init.orthogonal_(m.weight_hh_l0)
            init.zeros_(m.bias_ih_l0)
            init.zeros_(m.bias_hh_l0)
            if m.bidirectional:
                init.orthogonal_(m.weight_ih_l0_reverse)
                init.orthogonal_(m.weight_hh_l0_reverse)
                init.zeros_(m.bias_ih_l0_reverse)
                init.zeros_(m.bias_hh_l0_reverse)