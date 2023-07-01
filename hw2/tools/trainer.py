from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from tools.logger import Logger
import torch.multiprocessing
from model.config import CONFIG

torch.multiprocessing.set_sharing_strategy('file_system')

class AllConfig:
    model_name = ""
    task_name = ""
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-3
    max_length = 120
    model = CONFIG()

class Trainer:
    def __init__(self,optimizer,loss_func,scheduler,device,model,config:AllConfig):
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.device = device
        self.model = model
        self.config = config
        self.valid_stop_count = 0
        # valid_stop_count>5, stop train 
        self.best_valid={
            "acc":0.0,
            "loss":1.0,
            "epoch":0
        } 

    def train(self,dataloader,epoch_cnt):
        # 定义训练过程
        # 参数初始化
        if self.config.model.init_weight!="None":
            self.model.weight_init()
            
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        count, correct = 0, 0
        full_true = []
        full_pred = []
        # print("len(dataloader)",len(dataloader))
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (x, y) in loop:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss_func(output, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            correct += (output.argmax(1) == y).float().sum().item()
            count += len(x)
            full_true.append(y.cpu().numpy())
            full_pred.append(output.argmax(1).cpu().numpy())

            if batch_idx % 50 == 0:
                log_info = f'Train Epoch: {epoch_cnt} [{batch_idx * len(x)}/{len(dataloader)* len(x)} '\
                      + f'({round(100. * batch_idx / len(dataloader), 2)}%)]\tLoss: {loss.item()}'
                # print(log_info)
                Logger.log(self.config.task_name, log_info)

                # loop.set_description(f'[{task_ind + 1}/{task_all} {task_name}] Epoch [{epoch_cnt}/{epoch}]')
                # loop.set_postfix(loss=loss.item())

        train_loss *= self.config.batch_size    #将每个样本的平均损失转换为整个批次的总损失
        train_loss /= len(dataloader.dataset)
        train_acc = correct / count
        self.scheduler.step()
        f1 = f1_score(np.concatenate(full_true), np.concatenate(full_pred), average="binary")
        return train_loss, train_acc, f1


    def valid(self,dataloader,epoch_cnt):
        is_stop = False
        self.model.eval()
        # 验证过程
        val_loss, val_acc = 0.0, 0.0
        count, correct = 0, 0
        full_true = []
        full_pred = []
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss_func(output, y)
            val_loss += loss.item()
            correct += (output.argmax(1) == y).float().sum().item()
            count += len(x)
            full_true.append(y.cpu().numpy())
            full_pred.append(output.argmax(1).cpu().numpy())
        val_loss *= self.config.batch_size  # 将每个样本的平均损失转换为整个批次的总损失
        val_loss /= len(dataloader.dataset)
        val_acc = correct / count
        f1 = f1_score(np.concatenate(full_true), np.concatenate(full_pred), average="binary")
        # update best valid
        # print(val_acc)
        # print(self.best_valid["acc"])
        if val_acc>self.best_valid["acc"]:
            self.valid_stop_count = 0
            self.best_valid["acc"] = val_acc
            self.best_valid["loss"] = val_loss
            self.best_valid["epoch"] = epoch_cnt
        # valid_stop_count>5, stop train 
        else:
            self.valid_stop_count += 1
            if self.valid_stop_count>5:
                Logger.log(self.config.task_name,f"Stop At epoch {epoch_cnt}/{self.config.num_epochs} due to valid not increase")
                is_stop = True
        return val_loss, val_acc, f1, is_stop
    
    def test(self,dataloader,epoch_cnt):
        is_stop = False
        self.model.eval()
        # 验证过程
        test_loss, test_acc = 0.0, 0.0
        count, correct = 0, 0
        full_true = []
        full_pred = []
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss_func(output, y)
            test_loss += loss.item()
            correct += (output.argmax(1) == y).float().sum().item()
            count += len(x)
            full_true.append(y.cpu().numpy())
            full_pred.append(output.argmax(1).cpu().numpy())
        test_loss *= self.config.batch_size  # 将每个样本的平均损失转换为整个批次的总损失
        test_loss /= len(dataloader.dataset)
        test_acc = correct / count
        f1 = f1_score(np.concatenate(full_true), np.concatenate(full_pred), average="binary")
        return test_loss, test_acc, f1