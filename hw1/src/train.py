import json
import re
import os
from tqdm import tqdm
import argparse


class OneGrimDataloader:
    """
    一元模型词频表加载器
    """
    def __init__(self):
        self.word_dict = {}

    def count_grim_word(self, text: str):
        if len(text) < 1:
            return []
        for i in range(len(text)):
            if re.match(r'[\u4e00-\u9fa5]', text[i]):
                cur = text[i]
                if text[i] not in self.word_dict:
                    self.word_dict.update(
                        {cur: {"count": 0}})
                self.word_dict[cur]["count"] += 1

    def build_word_dict(self, path):
        # 遍历data目录下的所有语料库
        with open(path, 'r', encoding='gbk') as f:
            my_file = [line for line in f]
            for line in tqdm(my_file):
                self.count_grim_word(line)

    def save_dict(self, path="one_word_dict.json"):
        with open(path, 'w', encoding='gbk') as f:
            f.write(json.dumps(self.word_dict, ensure_ascii=False, indent=4))
        print(f"save dictionary in {path}")


class TwoGrimDataloader:
    """
    二元模型词频表加载器
    """
    def __init__(self):
        self.word_dict = {}

    def count_grim_word(self, text: str):
        if len(text) < 1:
            return []
        for i in range(len(text)):
            if re.match(r'[\u4e00-\u9fa5]', text[i]):
                bef = text[i - 1]
                cur = text[i]
                nex = text[i + 1]
                if text[i] not in self.word_dict:
                    self.word_dict.update(
                        {cur: {"begin": 0, "end": 0, "count": 0}})
                self.word_dict[cur]["count"] += 1

                if re.match(r"[\u4e00-\u9fa5]", nex):
                    if nex in self.word_dict[cur]:
                        self.word_dict[cur][nex] += 1
                    else:
                        self.word_dict[cur][nex] = 1
                elif re.match(r"[\u3000-\u303F]", nex):
                    self.word_dict[cur]["end"] += 1
                if re.match(r"[\u3000-\u303F]", bef):
                    self.word_dict[cur]["begin"] += 1

    def build_word_dict(self, path):
        # 遍历data目录下的所有语料库
        with open(path, 'r', encoding='gbk') as f:
            my_file = [line for line in f]
            for line in tqdm(my_file):
                self.count_grim_word(line)

    def save_dict(self,path="two_word_dict.json"):
        with open(path, 'w', encoding='gbk') as f:
            f.write(json.dumps(self.word_dict, ensure_ascii=False, indent=4))
        print(f"save dictionary in {path}")


class ThreeGrimDataloader:
    """
    三元模型词频表加载器
    """
    def __init__(self):
        self.word_dict = {}

    def count_grim_word(self, text: str):
        if len(text) < 3:
            return []
        for i in range(len(text)):
            if re.match(r'[\u4e00-\u9fa5]', text[i]):
                cur = text[i]
                nex = text[i + 1]
                nex_nex = text[i + 2]
                if text[i] not in self.word_dict:
                    self.word_dict.update(
                        {cur: {"count": 0}})
                if text[i] in self.word_dict:
                    self.word_dict[cur]["count"] += 1
                    if re.match(r"[\u4e00-\u9fa5][\u4e00-\u9fa5]", nex + nex_nex):
                        if nex in self.word_dict[cur]:
                            self.word_dict[cur][nex]["count"] += 1
                            if nex_nex in self.word_dict[cur][nex]:
                                self.word_dict[cur][nex][nex_nex] += 1
                            else:
                                self.word_dict[cur][nex].update({nex_nex: 1})
                        else:
                            self.word_dict[cur][nex] = {"count": 1, nex_nex: 1}

    def build_word_dict(self, path):
        with open(path, 'r', encoding='gbk') as f:
            my_file = [line for line in f]
            for line in tqdm(my_file):
                self.count_grim_word(line)

    def save_dict(self, path="three_word_dict.json"):
        with open(path, 'w', encoding='gbk') as f:
            f.write(json.dumps(self.word_dict, ensure_ascii=False, indent=4))
        print(f"save dictionary in {path}")


class FourGrimDataloader:
    """
    四元模型词频表加载器
    """
    def __init__(self):
        self.word_dict = {}

    def count_grim_word(self, text: str):
        if len(text) < 4:
            return []
        for i in range(len(text)):
            if re.match(r'[\u4e00-\u9fa5]', text[i]):
                cur = text[i]
                nex = text[i + 1]
                nex_nex = text[i + 2]
                nex_nex_nex = text[i + 3]
                if cur not in self.word_dict:
                    self.word_dict.update(
                        {cur: {"count": 0}})
                self.word_dict[cur]["count"] += 1
                if re.match(r"[\u4e00-\u9fa5]", nex):
                    if nex not in self.word_dict[cur]:
                        self.word_dict[cur].update({nex: {"count": 0}})
                    self.word_dict[cur][nex]["count"] += 1
                    if re.match(r"[\u4e00-\u9fa5]", nex_nex):
                        if nex_nex not in self.word_dict[cur][nex]:
                            self.word_dict[cur][nex].update({nex_nex: {"count": 0}})
                        self.word_dict[cur][nex][nex_nex]["count"] += 1
                        if re.match(r"[\u4e00-\u9fa5]", nex_nex_nex):
                            if nex_nex_nex not in self.word_dict[cur][nex][nex_nex]:
                                self.word_dict[cur][nex][nex_nex].update({nex_nex_nex: 0})
                            self.word_dict[cur][nex][nex_nex][nex_nex_nex] += 1

    def build_word_dict(self, path):
        with open(f"{path}", 'r', encoding='gbk') as f:
            my_file = [line for line in f]
            for line in tqdm(my_file):
                self.count_grim_word(line)

    def save_dict(self, path="four_word_dict.json"):
        with open(path, 'w', encoding='gbk') as f:
            f.write(json.dumps(self.word_dict, ensure_ascii=False, indent=4))
        print(f"save dictionary in {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--dict_type", type=str,
                        help="1,2,3,4")
    parser.add_argument("--path", type=str,
                        help="path to data")
    args = parser.parse_args()
    if args.dict_type == "1":
        dataloader = OneGrimDataloader()
        dataloader.build_word_dict(path=args.path)
        dataloader.save_dict()
    elif args.dict_type == "2":
        dataloader = TwoGrimDataloader()
        dataloader.build_word_dict(path=args.path)
        dataloader.save_dict()
    elif args.dict_type == "3":
        dataloader = ThreeGrimDataloader()
        dataloader.build_word_dict(path=args.path)
        dataloader.save_dict()

    elif args.dict_type == "4":
        dataloader = FourGrimDataloader()
        dataloader.build_word_dict(path=args.path)
        dataloader.save_dict()

    else:
        raise "no such type!"
