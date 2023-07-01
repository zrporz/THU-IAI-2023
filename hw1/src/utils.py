import json
from tqdm import tqdm
import os
import argparse


# 建立一个空字典，用于存储拼音到汉字的映射
def build_pinyin_word_dict(std_path='data/std_word_table.txt', pinyin2word_path='pinyin2word.txt'):
    pinyin_word_dict = {}
    std_word_list = []
    with open(std_path, 'r', encoding='gbk') as g:
        std_word_list = g.read()

    with open(pinyin2word_path, 'r', encoding='gbk') as f:
        # 打开一二级汉字对照表
        for line in f:
            # 用空格分割序列，得到一个列表
            pinyin_word_list = line.split()
            # 取出列表的第一个元素作为拼音
            pinyin = pinyin_word_list[0]
            # 取出列表的剩余元素作为汉字
            hanzi = pinyin_word_list[1:]
            std_hanzi = [word for word in hanzi if word in std_word_list]
            # 将拼音作为键，汉字作为值，添加到字典中
            pinyin_word_dict[pinyin] = std_hanzi
            print(std_hanzi)

    with open('model/pinyin2word/pinyin2word_dict.json', 'w', encoding='gbk') as f:
        f.write(json.dumps(pinyin_word_dict, ensure_ascii=False))


def sina_build_line(path):
    # 遍历指定目录下的所有语料库
    print(os.listdir(path))
    result = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(f"{path}/{filename}", 'r', encoding='gbk') as f:
                error = 0
                my_file = []
                for line in f:
                    try:
                        my_file.append(json.loads(line))
                    except:
                        error += 1

                for line in tqdm(my_file):
                    for _line in line['html'].split('\n'):
                        result.append("。。。" + _line + "。。。")
                    for _line in line['title'].split('\n'):
                        result.append("。。。" + _line + "。。。")
    with open("data/sina2016_build_line.txt", "w", encoding='gbk') as f:
        for line in result:
            if '\u2028' not in line:
                f.write(line+'\n')
            else:
                error+=1
    print("successfully write sina2016 data in  line in data/sina2016_build_line.txt")
    if error > 0:
        print(f"{error} lines occur")


def weibo_build_line(path):
    print(os.listdir(path))
    result = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(f"{path}/{filename}", 'r', encoding='gbk') as f:
                error = 0
                my_file = []
                for line in f:
                    try:
                        my_file.append(json.loads(line))
                    except:
                        error += 1

                for line in tqdm(my_file):
                    result.append("。。。" + line['content'] + "。。。")
    with open("data/weibo2020_build_line.txt", "w", encoding='gbk') as f:
        for line in result:
            f.write(line + '\n')
    print("successfully write weibo2020 data in  line in data/weibo2020_build_line.txt")
    if error > 0:
        print(f"{error} lines occur")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--data_type", type=str,
                        help="sina or sina data")
    parser.add_argument("--path", type=str,
                        help="path to data")
    args = parser.parse_args()
    # build_pinyin_word_dict()
    if args.data_type == "sina":
        sina_build_line(path=args.path)
    elif args.data_type == "sina":
        weibo_build_line(path=args.path)
