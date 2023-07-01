# README

## 项目结构

```
|-config_maker: 训练参数设置文件生成器
|-configs: 所有训练参数文件
|-data: 训练集、测试集、验证集
|-logs: 训练日志，Overall记录的是整体训练情况
|-model: 模型定义文件
|-tools
	|- trainer.py: 训练工具
	|- logger.py： 日志记录工具
|-README.md: 项目说明文件
```

## 运行说明

运行本项目需要安装pytorch，numpy，pathlib，gensim，json 等库。配置好环境后，运行 `python main.py -cf=<task\_name>` 即可运行实验，如果不指定配置文件则默认对 `configs` 下所有文件进行实验。
运行 `config_maker/<model_name>.py` 可批量生成配置文件