import torch.nn as nn
import torch
import torch.nn.functional as F
from model.config import CONFIG
import torch.nn.init as init


class RNN_LSTM(nn.Module):
    def __init__(self, config:CONFIG):

        super(RNN_LSTM, self).__init__()

        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        n_class = config.n_class
        self.__name__ = 'RNN_LSTM'

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        # (seq_len, batch, embed_dim) -> (seq_len, batch, 2 * hidden_size)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc1 = nn.Linear(64, n_class)

    def forward(self, inputs):
        _, (h_n, _) = self.encoder(self.embedding(inputs.to(torch.int64)).permute(1, 0, 2))  # (num_layers * 2, batch, hidden_size)
        # view h_n as (num_layers, num_directions, batch, hidden_size)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        return self.fc1(self.decoder(torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)))
    
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            if self.config.init_weight == "kaiming":
                init.kaiming_uniform_(m.weight)
            else:
                init.xavier_uniform_(m.weight)
            # or xavier_uniform_
            init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(param)
                elif 'bias' in name:
                    init.zeros_(param)
                    # set forget bias to 1
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)