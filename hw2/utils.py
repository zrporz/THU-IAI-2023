import gensim
import numpy as np
from collections import Counter
from typing import List, Dict
from pathlib import Path
import os

class CorpusLoader:
    def __init__(self, data_dir="./data", max_length=120):
        self.data_dir = Path(os.environ.get("DATA_DIR", data_dir))
        self.max_length = max_length
        self.word2id = None
        self.word2vecs = None

    def load_word2vec(self, vector_size=50):
        path = f"{self.data_dir}/wiki_word2vec_{vector_size}.bin"
        pre_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        self.word2vecs = np.array(np.zeros([len(self.word2id) + 1, pre_model.vector_size])) # add a zero vector for out of range word.
        for key in self.word2id:
            try:
                self.word2vecs[self.word2id[key]] = pre_model[key]
            except Exception:
                pass

    def get_word2id(self):
        word2id = Counter()
        for each in os.listdir(self.data_dir):
            if each.endswith('.txt'):
                with open(f"{self.data_dir}/{each}", encoding="utf-8") as f:
                    for line in f.readlines():
                        sentence = line.strip().split()
                        for word in sentence[1:]:
                            if word not in word2id.keys():
                                word2id[word] = len(word2id)
        self.word2id = word2id

    def load_corpus(self,path):
        contents, labels = np.array([0] * self.max_length), np.array([])
        with open(f"{self.data_dir}/{path}", encoding="utf-8", errors="ignore") as f:
            for line in f.readlines():
                sentence = line.strip().split()
                content = np.asarray([self.word2id.get(w, 0) for w in sentence[1:]])[:self.max_length]
                padding = max(self.max_length - len(content), 0)
                content = np.pad(content, ((0, padding)), "constant", constant_values=0)
                labels = np.append(labels, int(sentence[0]))
                contents = np.vstack([contents, content])
        print(f"Successfully load {len(contents)} data from {path}")
        contents = np.delete(contents, 0, axis=0)
        return contents, labels

    def load_all_data(self, train_file="train.txt", valid_file="valid.txt", test_file="test.txt", vector_size=50):
        files = [train_file, valid_file, test_file]
        self.get_word2id(files)
        self.load_word2vec(vector_size)
        train_data = self.load_corpus(train_file)
        valid_data = self.load_corpus(valid_file)
        test_data = self.load_corpus(test_file)
        return train_data, valid_data, test_data