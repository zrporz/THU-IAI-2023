import torch.nn as nn
import torch
import torch.nn.functional as F
from model.config import CONFIG
import torch.nn.init as init

class TextCNN(nn.Module):
    def __init__(self, config:CONFIG):
        super(TextCNN, self).__init__()
        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        n_class = config.n_class
        embedding_dim = config.embedding_dim
        kernel_num = config.kernel_num
        kernel_size = config.kernel_size
        drop_keep_prob = config.drop_keep_prob
        pretrained_embed = config.pretrained_embed

        self.__name__ = 'TextCNN'
        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        # 卷积层
        self.conv = [nn.Conv2d(1, kernel_num, (size, embedding_dim)) for size in kernel_size]
        # Dropout
        self.dropout = nn.Dropout(drop_keep_prob)
        # 全连接层
        self.fc = nn.Linear(len(kernel_size) * kernel_num, n_class)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x).squeeze(3))
        return F.max_pool1d(x, x.size(2)).squeeze(2)

    def forward(self, x):
        x = self.embedding(x.to(torch.int64)).unsqueeze(1)
        conv_list = []
        for _ in self.conv:
            conv_list.append(self.conv_and_pool(x, _))
        return F.log_softmax(self.fc(self.dropout(torch.cat(conv_list, dim=1))), dim=1)


    def weight_init(self,m):
        # 卷积层
        if isinstance(m, nn.Linear):
            if self.config.init_weight == "kaiming":
                init.kaiming_uniform_(m.weight)
            else:
                init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            if self.config.init_weight == "kaiming":
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                init.xavier_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            # or xavier_uniform_
            if m.bias is not None:
                init.zeros_(m.bias)