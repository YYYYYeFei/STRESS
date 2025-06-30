"""
Author: Fei Ye
Date: 2025-06
"""
import copy
import datetime
import numpy as np
import pandas as pd
import sys
import os
import torch.utils.data as data
from sklearn.model_selection import KFold, LeaveOneOut
from utils import cal_rmse,cal_pcc,cal_psnr,cal_ssim,cal_pcc01
sys.path.append("..")
from data import MyDataset
from utils import EarlyStopping
import torch
from torch import optim, nn
import model as module_arch

def train(args):
    """
    demo:python main.py --mode train --GPU_id 0 --tissue DLPFC --early_stopping
    # 如果数据集为Mus_MERFISH，则基因数为144
    :param args:
    :return:
    """
    args.data_root=args.data_root+f'/{args.tissue}/Input'
    sample_list=np.array(pd.read_csv(args.data_root+f'/{args.tissue}_samplelist.csv')['Sample'].tolist())
    loo=LeaveOneOut()
    loo.get_n_splits(sample_list)
    dir_root = f'./result/{args.tissue}/{args.model_name}'
    model_ft = getattr(module_arch, args.model_name)(args,in_chans=args.in_ch, gene_num=args.gene_num).to(args.device)
    initial_model = copy.deepcopy(model_ft)
    for i,(train_index,test_index) in enumerate(loo.split(sample_list)):
        if i>args.start_fold:
            print('\n########fold:{}'.format(i))
            model_save_root=os.path.join(dir_root,"Paras")
            os.makedirs(model_save_root,exist_ok=True)

            traindata = MyDataset(args, samples=sample_list[train_index])
            validdata = MyDataset(args, samples=sample_list[test_index])

            trainloader = torch.utils.data.DataLoader(
                dataset=traindata,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8
            )

            validloader = torch.utils.data.DataLoader(
                dataset=validdata,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8
            )
            model = copy.deepcopy(initial_model)
            model.train()
            loss = nn.MSELoss().to(args.device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            running_loss=0
            index_train=0
            history=[]
            loss_val_total=[]
            RMSEs=[]
            bestresult=10000
            PCCs=[]
            if args.early_stopping:
                print('INFO: Initializing early stopping')
                early_stopping = EarlyStopping(patience=args.patience)

            for epoch in range(args.num_epoch):
                print("\n")
                starttime = datetime.datetime.now()
                for t, data in enumerate(trainloader):
                    t_LR,t_HR = data
                    t_LR,t_HR=t_LR.to(args.device),t_HR.to(args.device)
                    result=model(t_LR)

                    optimizer.zero_grad()
                    l = loss(result, t_HR)
                    l.backward()
                    optimizer.step()
                    running_loss += l.item()
                    index_train += len(t_HR)

                averageLoss = running_loss / index_train
                history += [averageLoss]
                midtime = datetime.datetime.now()
                if (epoch + 1) % 100 == 0:
                    print('Epoch:[{}/{}]train loss:{:.4f}'.format(epoch + 1,args.num_epoch,averageLoss))
                    print("train time is {} seconds".format((midtime - starttime).seconds))

                # if ((epoch + 1) % 50 == 0) and (epoch != 0):
                #     print('Save epoch {} model parameters.....'.format(epoch))
                #     torch.save(model.state_dict(), model_save_root + "/epoch{}model.pth".format(epoch+1))

                model.eval()
                RMSE = []
                PCC = []
                index_val=0
                running_loss_val=0
                for v, v_data in enumerate(validloader):
                    v_LR,v_HR = v_data
                    v_LR,v_HR=v_LR.to(args.device),v_HR.to(args.device)

                    result_val = model(v_LR)
                    l_val = loss(result_val, v_HR)
                    index_val += len(result_val)
                    running_loss_val += l_val.item()

                    rmse = cal_rmse(v_HR, result_val)
                    pcc=cal_pcc01(v_HR.cpu().detach().numpy(), result_val.cpu().detach().numpy())

                    RMSE.append(rmse.cpu().detach().numpy())
                    PCC.append(pcc)

                averageLoss_test = running_loss_val / index_val
                ave_RSME = np.mean(RMSE)
                ave_PCC = np.mean(PCC)
                loss_val_total += [averageLoss_test]
                endtime = datetime.datetime.now()
                RMSEs += [np.mean(RMSE)]
                PCCs += [np.mean(PCC)]
                if (epoch + 1) % 1 == 0:
                    print('Epoch:[{}/{}] index_val:{} valid loss:{:.4f} RMSE:{} PCC:{}'.format(epoch + 1, args.num_epoch, index_val,
                                                                                              averageLoss_test, ave_RSME,ave_PCC))
                    print("test time is {} seconds".format((endtime - midtime).seconds))
                    print("Each epoch time is {} seconds".format((endtime - starttime).seconds))

                if bestresult > ave_RSME:
                    bestresult = ave_RSME
                    best_epoch=epoch
                    print("the best epoch is {}, the ave_RSME is {}".format(best_epoch,ave_RSME))
                    torch.save(model.state_dict(), model_save_root + f"/bestmodel_fold{i}.pth")

                if args.early_stopping:
                    early_stopping(averageLoss_test)
                    if early_stopping.early_stop:
                        print('===========early_stopping!!!!!!!!!!!')
                        break

            torch.save(model.state_dict(), model_save_root + f"/finalmodel_fold{i}.pth")
