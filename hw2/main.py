import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from utils import CorpusLoader
from model.RNN_LSTM import RNN_LSTM
from model.RNN_GRU import RNN_GRU
from model.TextCNN import TextCNN
from model.MLP import MLP
import argparse
import os
import configparser
from pathlib import Path
from tools.logger import Logger
import json
from tools.trainer import Trainer, AllConfig

corpusloader = CorpusLoader()

config = AllConfig()  # 配置模型参数

def LoadConfig(path:str):
    file_config = configparser.ConfigParser()
    file_config.read(Path.cwd() / 'configs' / path)
    
    config.model_name = file_config['model']['name']

    config.task_name = file_config['task']['task_name']

    config.batch_size = int(file_config['train']['batch_size'])
    config.num_epochs = int(file_config['train']['num_epochs'])
    config.learning_rate = float(file_config['train']['lr'])
    config.max_length = int(file_config['train']['max_length'])

    # 将加载的值更新到配置类中
    config.model.update_w2v = file_config.getboolean('model', 'update_w2v')
    config.model.drop_keep_prob = float(file_config['model']['drop_keep_prob'])

    if config.model_name == "TextCNN":
        config.model.kernel_num = int(file_config['model']['kernel_num'])
        config.model.kernel_size = list(map(int, file_config['model']['kernel_size'].split(',')))
    elif config.model_name == "MLP":
        config.model.hidden_dims = list(map(int, file_config['model']['hidden_dim'].split(',')))
    elif config.model_name == "RNN_GRU" or config.model_name == "RNN_LSTM":
        config.model.hidden_size = int(file_config['model']['hidden_size'])
        config.model.num_layers = int(file_config['model']['num_layers'])
    else:
        print(f"[{config.model_name}] is not included, please correctly choose neural network type")
        exit(1)

    if config.model_name == "RNN_LSTM":
        model = RNN_LSTM(config.model).to(DEVICE)
    elif config.model_name == "RNN_GRU":
        model = RNN_GRU(config.model).to(DEVICE)
    elif config.model_name == "TextCNN":
        model = TextCNN(config.model).to(DEVICE)
    elif config.model_name == "MLP":
        model = MLP(config.model).to(DEVICE)
    else:
        print(f"[{config.model_name}] is not included, please correctly choose neural network type")
        exit(1)
    return model
    

def load_data(max_length, batch_size):
    corpusloader.get_word2id()
    corpus_list = ["train.txt","validation.txt","test.txt"]
    dataloader_list = []    # train validation test
    for file in corpus_list:
        contents,labels = corpusloader.load_corpus(file)
        dataset = TensorDataset(torch.from_numpy(contents).type(torch.float),
                                torch.from_numpy(labels).type(torch.long))
        dataloader_list.append(DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)) 
    return dataloader_list[0], dataloader_list[1], dataloader_list[2]   # train validation test


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="IAI 2023_spring", description="Sentimental classification", allow_abbrev=True
    )
    parser.add_argument(
        "-cf",
        "--config_file",
        dest="config_file",
        type=str,
        default="all",
        help="choose configs under ./configs for train, [all] means train all configs under ./configs"
    )
    args = parser.parse_args()
    task_file_list = []
    if args.config_file == 'all':
        task_file_list = [file for file in os.listdir(Path.cwd() / 'configs') if not file.startswith('~')]
    else:
        task_file_list = [args.config_file]
    for config_file in task_file_list:
        print(f"Load task [{config_file}]. Task num: {task_file_list.index(config_file)+1}/{len(task_file_list)}")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # learning_rate, epoch, max_length, batch_size, neural_network, model = parser_data()
        model = LoadConfig(config_file)
        train_dataloader, val_dataloader, test_dataloader = load_data(config.max_length, config.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        # criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=5)
        trainer = Trainer(
            optimizer=optimizer,
            loss_func=nn.CrossEntropyLoss(),
            scheduler=scheduler,
            device=DEVICE,
            model=model,
            config=config
        )

        for each in tqdm(range(1, config.num_epochs + 1)):
            tr_loss, tr_acc, tr_f1 = trainer.train(train_dataloader,epoch_cnt=each)
            val_loss, val_acc, val_f1,is_stop = trainer.valid(val_dataloader,epoch_cnt=each)
            test_loss, test_acc, test_f1 = trainer.test(test_dataloader,epoch_cnt=each)
            result = {
                "tr_loss":tr_loss,
                "tr_acc":tr_acc,
                "tr_f1":tr_f1,
                "val_loss":val_loss,
                "val_acc":val_acc,
                "val_f1":val_f1,
                "test_loss":test_loss,
                "test_acc":test_acc,
                "test_f1":test_f1,
            }
            print(
                f"for epoch {each}, train_loss: {tr_loss:.4f}, train_acc: {tr_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}"
            )
            Logger.log(trainer.config.task_name,"========== Metrics ==========")
            Logger.log(trainer.config.task_name,json.dumps(result,indent=2))
            Logger.log(trainer.config.task_name,"========== Metrics ==========")
            if is_stop:
                # 是否需要停止  
                print(f"Stop At epoch {each}/{trainer.config.num_epochs} due to valid not incress")
                break
            
        Logger.log("Overall",f"[{config_file.split('.')[0]}] is finished at [epoch {each}/{trainer.config.num_epochs}], result:"+json.dumps(result,indent=2))
