import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CorpusLoader
from pathlib import Path

corpusloader = CorpusLoader()
corpusloader.get_word2id()
corpusloader.load_word2vec()

class CONFIG:
    update_w2v = True  # 是否在训练中更新w2v
    vocab_size = len(corpusloader.word2id) + 1  # 词汇量，与word2id中的词汇量一致
    init_weight = "None"    # None, xavier or kaiming, used for linear init, is not None, other layer will set their default init
    n_class = 2  # 分类数：分别为pos和neg
    embedding_dim = 50  # 词向量维度
    drop_keep_prob = 0.3  # dropout层，参数keep的比例
    kernel_num = 20  # 卷积层filter的数量
    kernel_size = [3, 5, 7]  # 卷积核的尺寸
    pretrained_embed = corpusloader.word2vecs  # 预训练的词嵌入模型
    hidden_size = 100  # 隐藏层神经元数
    num_layers = 2  # 隐藏层数
    hidden_dims = []

