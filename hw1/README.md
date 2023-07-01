# Input Method

对本项目原理、数据分析以及结论的更详细介绍可参考报告

## 项目结构及运行方法

本项目在python 3.10环境下运行，运行本程序需安装`json`、`re`、`os`、`tqdm`、`argpase`、`math`包，上交的项目文件结构如下：

```
|-data: 训练用的数据集
	|-input.txt：标准输入集
	|-output.txt：report中提到的最佳参数和模型下生成
	|-std_word_table.txt: 下发的拼音汉字对照表
	|-weibo2020_build_line.txt: 微博2020数据清洗后结果(新浪2016的数据清洗结果过大上传至网盘)
    |-sina：在三种测试集上新浪2016数据生成模型下的测试结果
    	|-500line: 标准测试集
    	|-poem: 诗歌测试集
    	|-semetic: 感情类文本测试集
    |-weibo：在三种测试集上微博2020数据生成模型下的测试结果
    	|-500line: 标准测试集
    	|-poem: 诗歌测试集
    	|-semetic: 感情类文本测试集
|-src
    |-model: 训练生成的模型（微博2020，新浪2016）
        |-pinyin2word.json: 拼音到汉字表处理成json形式
        |-std_word_table.txt: 一二级汉字表
        |-word_dictionary_weibo2020:微博2020生成的词频表
            |-one_word_dict.json: 一元词频表
            |-two_word_dict.json: 二元词频表
            |-three_word_dict.json: 三元词频表
            |-four_word_dict.json: 四元词频表
        |-word_dictionary_xinlang2016:新浪2016生成的词频表
            |-one_word_dict.json: 一元词频表
            |-two_word_dict.json: 二元词频表
            |-three_word_dict.json: 三元词频表（文件过大上传至云盘）
            |-four_word_dict.json: 四元词频表（文件过大上传至云盘）
    |-utils.py: 数据清理工具
    |-train.py: 训练生成模型
    |-viterbi.py: 隐式Markov模型主体算法
|-README.md: 项目说明文件
```

- 数据清洗：原始语料库以txt格式存储在 `\data\<语料库名称>` 文件夹下，运行`python utils.py --data_type=<sina or weibo> --path=<path to data>`进行数据清洗。
- 模型训练：运行`python train.py --data_type=<1/2/3/4> --path=<path to data>`进行训练。生成的词频表`<one/two/three/four>_word_dict.json`存储在根目录下。
- 拼音输入法：`python viterbi.py --mode=<run/test> --model_path=<path to model> --test_path=<path to test dataset>`

## 模型结构

以二元词表为例，词频表格式为：

```
{"前置字":{"count":xxx,"begin":xxx,"end":xxx,"后置字1":xxx,"后置字2":xxx,"后置字3":xxx}...
}
```

每个字段含义如下：

- 前置字：二元字组中出现在第一个位置的字
- 后置字：二元组出现在第二个位置的字
- count：前置字出现的总次数
- begin：前置字出现在一个分句的开头的次数（定义一个分句为由中文标点符号分隔的句子）
- end：前置字出现在一个分局末尾的次数

## 主要结论

- 更多元的模型会带来更好的效果，但同时也会带来更高的模型训练和运行成本，且在不改变模型算法的条件下，仅靠增加模型元数带来的边际收益递减，甚至会带来负收益。
- 数学模型仅仅为算法提供了理论依据，在不违背数学理论的大前提下，一些技巧如平滑化、加权平均等策略会起到很好的效果。
- 数据清洗十分重要，在数据清洗时尝试过滤掉更多杂质信息，对模型性能提升很有帮助
- 语料库规模和类型对于模型准确率至关重要，选取和任务适配的语料库类型是一种性价比极高的策略。