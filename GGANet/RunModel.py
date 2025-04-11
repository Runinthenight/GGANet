# -*- coding:utf-8 -*-

import os
import random
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
import dgl
import pickle as pkl
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from .config import config
from .model import GeoDTI
from .utils.DataPrepare import get_kfold_data, shuffle_dataset
from .utils.DatasetFunction import CustomDataSet, collate_fn
from .utils.EarlyStoping import EarlyStopping
from .LossFunction import CELoss, PolyLoss
from matplotlib import pyplot as plt 


class Trainer:
    

    def __init__(self, model, config, exp_name, dataset, save_dir="./output", watch_time=False):
        self.hp = config
        self.watch_time = watch_time
        self.hp.lr = config.lr
        self.save_dir = os.path.join(save_dir, exp_name, dataset)
        self.model = model(config)
        self.protein_embeding_dir = self.hp.protein_embedding_path
        self.drug_embeding_dir = self.hp.drug_embedding_path
        assert dataset in ["DrugBank", "KIBA",  "Enzyme", "GPCRs", "ion_channel"]
        self.dataset = dataset
        self.dti_dir = self.hp.dti_path
        
        random.seed(self.hp.seed)
        torch.manual_seed(self.hp.seed)
        torch.cuda.manual_seed_all(self.hp.seed)
        pass

    def train(self):
        train_pbar = tqdm(
                enumerate(BackgroundGenerator(self.train_dataset_loader)),
                total=len(self.train_dataset_loader))

        """train"""
        train_losses_in_epoch = []
        self.model.train()
        for train_i, train_data in train_pbar:
            t1 = time.time()
            (train_compounds, train_protein, train_labels) = train_data
            t2= time.time()
            if self.watch_time:
                print(f"数据加载时间：{t2-t1}s")
            self.optimizer.zero_grad()

            predicted_interaction, simLoss = self.model(train_compounds, train_protein)
            t3 = time.time()
            if self.watch_time:
                print(f"模型计算时间：{t3-t2}s")
            train_loss = self.Loss(predicted_interaction, train_labels) * 0.5 + simLoss *0.5
            train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            t4 = time.time()
            if self.watch_time:
                print(f"梯度下降时间：{t4 - t3}s")
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  # 一次epoch的平均训练loss
        return train_loss_a_epoch

    def valid(self, loader,dataset_class, save=False, show_weights=False):
        valid_losses_in_epoch = []
        self.model.eval()
        data_pbar = tqdm(
                enumerate(BackgroundGenerator(loader)),
                total=len(loader))
        Y, P, S = [], [], []
        with torch.no_grad():
            for valid_i, valid_data in data_pbar:

                valid_compounds, valid_proteins, valid_labels = valid_data
                valid_scores, SimLoss = self.model(valid_compounds, valid_proteins)
                valid_loss = self.Loss(valid_scores, valid_labels)*0.5 + SimLoss*0.5
                valid_losses_in_epoch.append(valid_loss.item())
                valid_labels = valid_labels.to('cpu').data.numpy()
                valid_scores = F.softmax(
                    valid_scores, 1).to('cpu').data.numpy()
                valid_predictions = np.argmax(valid_scores, axis=1)
                valid_scores = valid_scores[:, 1]

                Y.extend(valid_labels)
                P.extend(valid_predictions)
                S.extend(valid_scores)

        Precision_dev = precision_score(Y, P)
        Reacll_dev = recall_score(Y, P)
        Accuracy_dev = accuracy_score(Y, P)
        AUC_dev = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC_dev = auc(fpr, tpr)
        loss_a_epoch = np.average(valid_losses_in_epoch)
        if save:
            results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
            .format(dataset_class, loss_a_epoch, Accuracy_dev,Precision_dev,  Reacll_dev, AUC_dev, PRC_dev, )
            with open(self.fold_save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
                f.write("Test the stable model" + '\n')
                f.write(results + '\n')
            filepath = os.path.join(self.fold_save_path ,f"{dataset_class}_prediction.txt")
            if os.path.exists(filepath):
                os.remove(filepath) 
            with open(filepath, 'a') as f:
                f.write("target" + " " + "predict" + " " + "score"'\n')
                for i in range(len(Y)):
                    f.write(f"{Y[i]} {P[i]} {S[i]:.2f}"  +'\n')
        return loss_a_epoch, Accuracy_dev, AUC_dev, PRC_dev, Precision_dev, Reacll_dev

    def run(self):
        self.split_data()
        Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
        
        for i_fold in range(self.hp.fold_nums):
            self.init_model()
            self.init_draw_data()
            self.set_loader(i_fold)
            self.set_optimizer()
            early_stopping = EarlyStopping(savepath=self.fold_save_path, patience=self.hp.patience, verbose=True, delta=0)
            for i in range(self.hp.epoch):
                train_loss_a_epoch = self.train()
                valid_loss_a_epoch, Accuracy_dev, AUC_dev, PRC_dev, Precision_dev, Reacll_dev = self.valid(self.valid_dataset_loader, "Valid")
                self.draw_train_loss.append(train_loss_a_epoch)
                self.draw_valid_loss.append(valid_loss_a_epoch)
                self.draw_valid_acc.append(Accuracy_dev)
                self.draw_valid_auc.append(AUC_dev)
                self.draw_valid_aupr.append(PRC_dev)
                epoch_len = len(str(self.hp.epoch))
                print_msg = (f'[{i:>{epoch_len}}/{self.hp.epoch:>{epoch_len}}] ' +
                            f'train_loss: {train_loss_a_epoch:.5f} ' +
                            f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                            f'valid_AUC: {AUC_dev:.5f} ' +
                            f'valid_PRC: {PRC_dev:.5f} ' +
                            f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                            f'valid_Precision: {Precision_dev:.5f}' +                                                                             
                            f'valid_Reacll: {Reacll_dev:.5f} ')
                print(print_msg)
                early_stopping(Accuracy_dev, self.model, self.hp.epoch)
                if early_stopping.early_stop:
                    break
            self.draw("train_process.png")

            self.model.load_state_dict(torch.load(self.fold_save_path + '/valid_best_checkpoint.pth'))
            train_loss_a_epoch, Accuracy_train, AUC_train, PRC_train, Precision_train, Recall_train= self.valid(self.train_dataset_loader, "Train", True)
            valid_loss_a_epoch, Accuracy_valid, AUC_valid, PRC_valid, Precision_valid, Recall_valid= self.valid(self.valid_dataset_loader, "Valid", True)
            test_loss_a_epoch, Accuracy_test, AUC_test, PRC_test, Precision_test, Recall_test= self.valid(self.test_dataset_loader, "Test", True)
            Accuracy_List_stable.append(Accuracy_test)
            AUC_List_stable.append(AUC_test)
            AUPR_List_stable.append(PRC_test)
            Recall_List_stable.append(Recall_test)
            Precision_List_stable.append(Precision_test)
        self.show_result(Accuracy_List_stable, Precision_List_stable, Recall_List_stable, AUC_List_stable, AUPR_List_stable)

    def draw(self, pic_name):
        x = list(range(1, len(self.draw_valid_aupr)+1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 行 2 列

        # 左图
        axes[0].plot(x, self.draw_train_loss, marker='o', linestyle='-', color='b', label='train')
        axes[0].plot(x, self.draw_valid_loss, marker='o', linestyle='-', color='r', label='valid')
        axes[0].set_title('loss')
        axes[0].set_xlabel('loss')
        axes[0].set_ylabel('epoch')
        axes[0].legend()
        axes[0].grid(True)

        # 右图
        axes[1].plot(x, self.draw_valid_acc, marker='o', linestyle='--', color='r', label='ACC')
        axes[1].plot(x, self.draw_valid_auc, marker='o', linestyle='--', color='b', label='AUC')
        axes[1].plot(x, self.draw_valid_aupr, marker='o', linestyle='--', color='g', label='AUPR')
        axes[1].set_title('metrics')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend()
        axes[1].grid(True)

        # 调整布局并显示
        plt.tight_layout()
        plt.savefig(os.path.join(self.fold_save_path, pic_name))
        plt.close()

    def split_data(self):
        print("Train in " + self.dataset)
        print("load data")

        dti_path = f'{self.dti_dir}/{self.dataset}.txt'
        with open(dti_path, "r") as f:
            data_list = f.read().strip().split('\n')
        print("load finished")

        print("data shuffle")
        data_list = shuffle_dataset(data_list, self.hp.seed)

        '''split dataset to train&validation set and test set'''
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        self.train_data_list = data_list[0:split_pos]
        self.test_data_list = data_list[split_pos:-1]
        print('Number of Train&Val set: {}'.format(len(self. train_data_list)))
        print('Number of Test set: {}'.format(len(self. test_data_list)))
        protein_path = os.path.join(self.protein_embeding_dir, self.dataset)
        drug_path = os.path.join(self.drug_embeding_dir, self.dataset)
        test_dataset = CustomDataSet(self.test_data_list, drug_path, protein_path)
        self.test_dataset_loader = DataLoader(test_dataset, batch_size=self.hp.batch_size, shuffle=False, num_workers=0,
                                         collate_fn=collate_fn)
        
    def set_loader(self, i_fold):
        self.fold_save_path = os.path.join(self.save_dir, str(i_fold))
        if not os.path.exists(self.fold_save_path):
            os.makedirs(self.fold_save_path)
        train_dataset, valid_dataset = get_kfold_data(
            i_fold, self.train_data_list, k=self.hp.fold_nums)
        protein_path = os.path.join(self.protein_embeding_dir, self.dataset)
        drug_path = os.path.join(self.drug_embeding_dir, self.dataset)
        train_dataset = CustomDataSet(train_dataset, drug_path, protein_path)
        valid_dataset = CustomDataSet(valid_dataset, drug_path, protein_path)
        
        self.train_dataset_loader = DataLoader(train_dataset, batch_size=self.hp.batch_size, shuffle=True, num_workers=0,
                                          collate_fn=collate_fn)
        self.valid_dataset_loader = DataLoader(valid_dataset, batch_size=self.hp.batch_size, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn )
    
    def init_model(self, model_path = None):
        self.model = self.model.to(self.hp.device)
       
        if model_path:
            print(f"Load weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        else:
            print("""Initialize weights""")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)

    def init_draw_data(self):
        self.draw_train_loss = []
        self.draw_valid_loss = []
        self.draw_valid_acc = []
        self.draw_valid_auc = []
        self.draw_valid_aupr = []

    def show_result(self, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
        Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.std(Accuracy_List)
        Precision_mean, Precision_var = np.mean(
            Precision_List), np.std(Precision_List)
        Recall_mean, Recall_var = np.mean(Recall_List), np.std(Recall_List)
        AUC_mean, AUC_var = np.mean(AUC_List), np.std(AUC_List)
        PRC_mean, PRC_var = np.mean(AUPR_List), np.std(AUPR_List)
        print("The model's results:")
        filepath = os.path.join(self.save_dir, "result.txt")
        with open(filepath, 'w') as f:
            f.write('Accuracy(std):{:.4f}({:.4f})'.format(
                Accuracy_mean, Accuracy_var) + '\n')
            f.write('Precision(std):{:.4f}({:.4f})'.format(
                Precision_mean, Precision_var) + '\n')
            f.write('Recall(std):{:.4f}({:.4f})'.format(
                Recall_mean, Recall_var) + '\n')
            f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
            f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
        print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
        print('Precision(std):{:.4f}({:.4f})'.format(
            Precision_mean, Precision_var))
        print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
        print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
        print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

    def set_optimizer(self, ):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hp.lr)
        self. scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.hp.lr, max_lr=self.hp.lr*2, cycle_momentum=False,
                                                step_size_up=len(self. train_dataset_loader))  
        self.Loss = CELoss(device=self.hp.device)  
    
    def test_dti(self, drug_id, protein_id, model_path):
        self.init_model(model_path)
        dti_path = f'{self.dti_dir}/{self.dataset}.txt'
        all_data_df = pd.read_csv(dti_path, header=None, dtype=str)
        dti = all_data_df[(all_data_df.iloc[:,0]==drug_id) & (all_data_df.iloc[:,1]==protein_id)].iloc[0]
        dti_str = ",".join(dti)
        protein_path = os.path.join(self.protein_embeding_dir, self.dataset)
        drug_path = os.path.join(self.drug_embeding_dir, self.dataset)
        dataset = CustomDataSet([dti_str], drug_path, protein_path)
        loader = DataLoader(dataset, collate_fn=collate_fn)
        predict, sim_loss, weights = self.valid(loader, "test", show_weights=True)





# if __name__ == "__main__":
    # print(os.path.dirname(os.path.abspath(__file__)))
    # exp_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    # hp = config()
    
    # for dataset in  ["DrugBank"]:
    #     trainer = Trainer(GeoDTI, hp, exp_name, dataset,watch_time=False)
    #     trainer.run()

    