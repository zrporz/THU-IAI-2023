import json
import math
from tqdm import tqdm
import argparse


def prob2distance(prob: float):
    """
    将概率取负对数后返回，如果概率为0则返回无穷
    """
    if prob == 0:
        return float("inf")
    else:
        return -math.log(prob)


class HMM:
    def __init__(self,model_path:str,_coff=[0.0, 1.61, 7.8, 0.0], _min_prob=float("1e-6"),):
        """
        :param model_path:词典存储的路径，如 model/word_dictionary_weibo2020
        :param _coff:存储一元、二元、三元、四元概率在推算条件概率时的占比
        :param _min_prob:对于不存在于词典中的值，为其赋予一个特定的最小概率值
        """
        self.one_word_dict = {}
        self.two_word_dict = {}
        self.three_word_dict = {}
        self.four_word_dict = {}
        self.pinyin2word_dict = {}
        self.distance = []
        self.pinyin_list = ""
        self.coff = _coff
        self.cur_sum = 0
        self.min_prob = _min_prob
        self.model_path = model_path

    def set_cov(self, coff, _min_prob):
        self.coff = coff
        self.min_prob = _min_prob

    def load_one_word_dict(self):
        """
        将词频表加载至 self.one_word_dict
        """
        word_count = 0
        path = f"{self.model_path}/one_word_dict.json"
        with open(path, 'r', encoding='gbk') as f:
            self.one_word_dict = json.load(f)
        for word in self.one_word_dict:
            word_count += self.one_word_dict[word]["count"]
        print(f"load {word_count} words pair from one_word_dict")

    def load_two_word_dict(self):
        """
        将词频表加载至 self.two_word_dict
        """
        word_count = 0
        path = f"{self.model_path}/two_word_dict.json"
        with open(path, 'r', encoding='gbk') as _f:
            self.two_word_dict = json.load(_f)
        for word in self.two_word_dict:
            word_count += self.two_word_dict[word]["count"]
        print(f"load {word_count} words pair from two_word_dict")

    def load_three_word_dict(self):
        """
        将词频表加载至 self.three_word_dict
        """
        word_count = 0
        path = f"{self.model_path}/three_word_dict.json"
        with open(path, 'r', encoding='gbk') as f:
            self.three_word_dict = json.load(f)
        for word in self.three_word_dict:
            word_count += self.three_word_dict[word]["count"]
        print(f"load {word_count} words pair from three_word_dict")

    def load_four_word_dict(self):
        """
        将词频表加载至 self.four_word_dict
        """
        path = f"{self.model_path}/four_word_dict.json"
        word_count = 0
        with open(path, 'r', encoding='gbk') as f:
            self.four_word_dict = json.load(f)
        for word in self.four_word_dict:
            word_count += self.four_word_dict[word]["count"]
        print(f"load {word_count} words pair from four_word_dict")

    def load_pinyin2word_dict(self, path="model/pinyin2word/pinyin2word_dict.json"):
        """
        将拼音-汉字表加载至 self.pinyin2word_dict
        """
        with open(path, 'r', encoding='gbk') as f:
            self.pinyin2word_dict = json.load(f)
        print("successfully load pinyin2word_dict!")

    def prob_one_word(self, cur: str):
        """
        求一元词表中汉字出现的频率
        :param cur: 给定汉字
        :return: 返回频率，如果为0则返回一个最小概率
        """
        try:
            return self.one_word_dict[cur]["count"] / self.cur_sum
        except:
            return self.min_prob

    def prob_two_word(self, last: str, cur: str):
        """
        求二元词表中汉字出现的频率
        :param  last: 上一个
        :param  cur: 当前汉字
        :return: 返回频率，如果为0则返回一个最小概率
        """
        try:
            return self.two_word_dict[last][cur] / self.two_word_dict[last]["count"]
        except:
            return self.min_prob

    def prob_three_word(self, last_last: str, last: str, cur: str):
        """
        求三元词表中汉字出现的频率
        :param last_last: 上上个
        :param last: 上一个
        :param cur: 当前汉字
        :return: 返回频率，如果为0则返回一个最小概率
        """
        try:
            return self.three_word_dict[last_last][last][cur] / self.three_word_dict[last_last][last]["count"]
        except:
            return self.min_prob

    def prob_four_word(self, last_last_last: str, last_last: str, last: str, cur: str):
        """
                求四元词表中汉字出现的频率
                :param last_last_last: 上上上个
                :param last_last: 上上个
                :param last: 上一个
                :param cur: 当前汉字
                :return: 返回频率，如果为0则返回一个最小概率
                """
        try:
            return self.four_word_dict[last_last_last][last_last][last][cur] / \
                   self.four_word_dict[last_last_last][last_last][last]["count"]
        except:
            return self.min_prob

    def sum_begin(self, pinyin: str):
        """
        求某一个拼音作为一段话开头的所有字之和
        :param pinyin: 给定拼音
        :return: 该拼音所有汉字作为开头的频率求和
        """
        sum_ = 0
        for cur in self.pinyin2word_dict[pinyin]:
            if cur in self.two_word_dict:
                sum_ += self.two_word_dict[cur]["begin"]
        return sum_

    def sum_count_one(self, pinyin: str):
        """
        求一元词表中某一个拼音对应所有字的个数
        :param pinyin: 给定拼音
        :return: 该拼音所有出现频率求和
        """
        sum_ = 0
        for cur in self.pinyin2word_dict[pinyin]:
            if cur in self.one_word_dict:
                sum_ += self.one_word_dict[cur]["count"]
        return sum_

    def sum_count_two(self, pinyin: str):
        """
        求二元词表中某一个拼音对应所有字的个数
        :param pinyin: 给定拼音
        :return: 该拼音所有出现频率求和
        """
        sum_ = 0
        for cur in self.pinyin2word_dict[pinyin]:
            if cur in self.two_word_dict:
                sum_ += self.two_word_dict[cur]["count"]
        return sum_

    def begin_strategy(self, _ind: int):
        """
        首字策略
        :param _ind: 一般为0
        :return: None
        """
        pinyin = self.pinyin_list[_ind]
        begin_count = self.sum_begin(pinyin)
        for cur in self.pinyin2word_dict[pinyin]:
            # 目前 one_grim 策略只应用于首字符，因此before统一设为None
            if cur in self.two_word_dict:
                self.distance[_ind][cur] = {
                    "dist": prob2distance(self.two_word_dict[cur]["begin"] / begin_count), "before": None}
            else:
                self.distance[_ind][cur] = {
                    "dist": float("inf"), "before": None}

    def two_grin_strategy(self, _ind: int):
        """
        二元策略
        :param _ind: 转移概率矩阵中的下标
        :return: None
        """
        pinyin = self.pinyin_list[_ind]
        last_pinyin = self.pinyin_list[_ind - 1]
        for cur in self.pinyin2word_dict[pinyin]:
            self.distance[_ind][cur] = {"dist": float(
                "inf"), "before": self.pinyin2word_dict[last_pinyin][0]}
            for last in self.pinyin2word_dict[last_pinyin]:
                ans_dist = self.distance[_ind - 1][last]['dist'] \
                           + prob2distance(self.coff[1] * self.prob_two_word(last, cur) \
                                           + self.coff[0] * self.prob_one_word(cur))
                if ans_dist < self.distance[_ind][cur]["dist"]:
                    self.distance[_ind][cur]["dist"] = ans_dist
                    self.distance[_ind][cur]["before"] = last

    def three_grin_strategy(self, _ind: int):
        """
        三元策略
        :param _ind: 转移概率矩阵中的下标
        :return: None
        """
        pinyin = self.pinyin_list[_ind]
        last_pinyin = self.pinyin_list[_ind - 1]
        self.cur_sum = self.sum_count_one(pinyin)
        for cur in self.pinyin2word_dict[pinyin]:
            self.distance[_ind][cur] = {"dist": float(
                "inf"), "before": self.pinyin2word_dict[last_pinyin][0]}
            for last in self.pinyin2word_dict[last_pinyin]:
                last_last = self.distance[_ind - 1][last]['before']
                ans_dist = self.distance[_ind - 1][last]['dist'] \
                           + prob2distance(self.coff[2] * self.prob_three_word(last_last, last, cur) \
                                           + self.coff[1] * self.prob_two_word(last, cur) \
                                           + self.coff[0] * self.prob_one_word(cur))

                if ans_dist < self.distance[_ind][cur]["dist"]:
                    self.distance[_ind][cur]["dist"] = ans_dist
                    self.distance[_ind][cur]["before"] = last

    def four_grin_strategy(self, _ind: int):
        """
        四元策略
        :param _ind: 转移概率矩阵中的下标
        :return: None
        """
        pinyin = self.pinyin_list[_ind]
        last_pinyin = self.pinyin_list[_ind - 1]
        self.cur_sum = self.sum_count_one(pinyin)
        for cur in self.pinyin2word_dict[pinyin]:
            self.distance[_ind][cur] = {"dist": float(
                "inf"), "before": self.pinyin2word_dict[last_pinyin][0]}
            for last in self.pinyin2word_dict[last_pinyin]:
                last_last = self.distance[_ind - 1][last]['before']
                last_last_last = self.distance[_ind - 2][last_last]['before']
                ans_dist = self.distance[_ind - 1][last]['dist'] \
                           + prob2distance(self.coff[3] * self.prob_four_word(last_last_last, last_last, last, cur) \
                                           + self.coff[2] * self.prob_three_word(last_last, last, cur) \
                                           + self.coff[1] * self.prob_two_word(last, cur) \
                                           + self.coff[0] * self.prob_one_word(cur))

                if ans_dist < self.distance[_ind][cur]["dist"]:
                    self.distance[_ind][cur]["dist"] = ans_dist
                    self.distance[_ind][cur]["before"] = last

    def determine_shortest_way(self):
        """
        对尾部进行一次条件概率判断
        :return: 返回距离最小（概率最大）的结点
        """
        key_end = ""
        min_dist = float("inf")
        for key in self.distance[-1]:
            last_dist = 0
            try:
                last_dist = prob2distance(self.two_word_dict[key]["end"] / self.two_word_dict[key]["count"])
            except:
                last_dist = float("inf")
            if min_dist >= self.distance[-1].get(key)["dist"] + last_dist:
                min_dist = self.distance[-1].get(key)["dist"] + last_dist
                key_end = key
        return key_end

    def infer(self, text: str):
        """
        HMM推断过程
        :param text: 给定拼音串
        :return: 翻译汉字字符串结果
        """
        self.distance = []
        self.pinyin_list = text.strip('\n').split(' ')
        # 对 pinyin_list 线性扫描推断，概率计算的原则是只要前面有足够的拼音个数，就采用尽可能多的n元策略
        for _ind in range(len(self.pinyin_list)):
            self.distance.append({})
            if _ind == 0:
                # 如果时第一个拼音，则采用首字符策略
                self.begin_strategy(_ind)
            elif _ind == 1:
                # 如果是第二个拼音，则采用二元策略
                self.two_grin_strategy(_ind)
            elif _ind == 2:
                # 如果是第三个拼音，则采用三元策略
                self.three_grin_strategy(_ind)
            else:
                # 大于等于第四个拼音，采用四元策略
                self.four_grin_strategy(_ind)
        # 进行一步尾部判断
        key_end = self.determine_shortest_way()
        output = key_end
        cur = key_end
        _ind = len(self.pinyin_list) - 1
        while _ind > 0:
            cur = self.distance[_ind][cur]["before"]
            output += cur
            _ind -= 1
        return output[::-1]


def test_accuracy(path):
    std_output = []
    output = []
    with open(f'{path}/std_output.txt', 'r', encoding='utf-8') as f:
        std_output = [line for line in f]
    with open(f'{path}/output.txt', 'r', encoding='gbk') as f:
        output = [line for line in f]
    correct_line = 0
    correct_word = 0
    word_sum = 0
    line_sum = 0
    error_example = []
    for _ix in range(len(std_output)):
        word_sum += len(std_output[_ix])
        line_sum += 1
        if std_output[_ix] == output[_ix]:
            correct_line += 1
            correct_word += len(std_output[_ix])
        else:
            count = 0
            for word in range(len(std_output[_ix])):
                if std_output[_ix][word] == output[_ix][word]:
                    count += 1
            error_example.append([std_output[_ix], output[_ix]])
            correct_word += count
    print(f"line accuracy: {correct_line / line_sum * 100}%")
    print(f"word accuracy: {correct_word / word_sum * 100}%")
    print(error_example[0:min(5, len(error_example))])
    return correct_line / line_sum, correct_word / word_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--mode", type=str,
                        help="run or test mode")
    parser.add_argument("--model_path", type=str,
                        help="model dictionary")
    parser.add_argument("--test_path", type=str, default="test_dataset/500line",
                        help="model dictionary")
    args = parser.parse_args()
    hmm = HMM(model_path=args.model_path)
    hmm.load_pinyin2word_dict()
    hmm.load_one_word_dict()
    hmm.load_two_word_dict()
    hmm.load_three_word_dict()
    hmm.load_four_word_dict()
    print(f"running in {args.mode} mode")
    if args.mode == "run":
        # run模式，进行拼音到汉字的翻译
        while True:
            s = input()
            print(hmm.infer(s))
    if args.mode == "test":
        # test模式，指定测试集后测量准确率并输出至指定路径
        # _coff=[0.0, 1.61, 7.8, 0.0] min_prob = float("1e-6")
        min_prob = [float("1e-6")]
        coff_0 = [0.0]
        coff_1 = [1.61]
        coff_2 = [7.8]
        coff_3 = [0]
        ind = 0
        with open("result/test_result.txt", "w") as res:
            for x in coff_0:
                for y in coff_1:
                    for z in coff_2:
                        for s in coff_3:
                            for k in min_prob:
                                print(
                                    f"testing in coff_0 = {x}, coff_1 = {y}, coff_2 = {z}, coff_3 = {s}, min_prob = {k}")
                                hmm.set_cov(coff=[x, y, z, s], _min_prob=k)
                                with open(f'{args.test_path}/input.txt', 'r') as f:
                                    input_list = [line for line in f]
                                output_list = []
                                for line in tqdm(input_list):
                                    output_list.append(hmm.infer(line))
                                with open(f'{args.test_path}/output.txt', 'w', encoding='gbk') as f:
                                    for ix in range(len(output_list)):
                                        f.write(output_list[ix] + '\n')
                                _1, _2 = test_accuracy(path=args.test_path)
                                res.write(
                                    f"coff_0:{x}, coff_1:{y}, coff_2:{z}, coff_3 = {s}, min_prob = {k} \n line accuracy:{_1:.4f}, word accuracy:{_2:.4f}\n")
        print("test over, result save in ./result/test_result.txt")
