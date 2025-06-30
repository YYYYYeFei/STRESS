"""
Author: Fei Ye
Date: 2025-06
"""
import numpy as np
import pandas as pd
import sys
import os
import torch.utils.data as data
from sklearn.model_selection import LeaveOneOut
from utils import cal_rmse,cal_psnr,cal_ssim,cal_pcc,cal_pcc01
sys.path.append("..")
from data import MyDataset
import torch
import model as module_arch

def test(args):
    print('=' * 20 + 'Valid information' + '=' * 20)
    args.data_root=args.data_root+f'/{args.tissue}/Input'
    sample_list=np.array(pd.read_csv(args.data_root+f'/{args.tissue}_samplelist.csv')['Sample'].tolist())
    loo=LeaveOneOut()
    loo.get_n_splits(sample_list)
    dir_root = f'./result/{args.tissue}/{args.model_name}/'
    loo_PCC=[]
    loo_rmse = []
    loo_PSNR=[]
    loo_SSIM = []
    for i,(_,test_index) in enumerate(loo.split(sample_list)):
        if i > -1:
            model_save_root=os.path.join(dir_root,'Paras')
            validdata = MyDataset(args, samples=sample_list[test_index])
            validloader = torch.utils.data.DataLoader(
                dataset=validdata,
                batch_size=1,
                shuffle=False,
                num_workers=8
            )
            model = getattr(module_arch, args.model_name)(args,in_chans=args.in_ch,gene_num=args.gene_num).to(args.device)
            model.load_state_dict(torch.load(model_save_root + f"/finalmodel_fold{i}.pth"), False)
            model.eval()
            RMSE = []
            PCC = []
            PSNR=[]
            SSIM=[]
            index_val=0
            for v, v_data in enumerate(validloader):
                v_LR, v_HR = v_data
                file=sample_list[test_index]
                v_LR,v_HR=v_LR.to(args.device),v_HR.to(args.device)
                v_HR=v_HR.squeeze(1)
                result_val = model(v_LR)
                index_val += len(result_val)
                # print(f'v_HR.shape:{v_HR.shape},result_val.shape:{result_val.shape}')

                rmse = cal_rmse(v_HR.cpu().detach(), result_val.cpu().detach())
                psnr = cal_psnr(v_HR.cpu().detach(), result_val.cpu().detach())
                ssim = cal_ssim(v_HR.cpu().detach(), result_val.cpu().detach())
                pcc = cal_pcc01(v_HR.cpu().detach().numpy(), result_val.cpu().detach().numpy())
                RMSE += [rmse.cpu().detach().numpy()]
                PSNR += [psnr.cpu().detach().numpy()]
                SSIM += [ssim.cpu().detach().numpy()]
                PCC += [pcc]

                save_root = dir_root + '/Test/Output'
                os.makedirs(save_root, exist_ok=True)
                np.save(save_root + '/{}_LR_fold{}.npy'.format(file[0],i), v_LR.cpu().detach().numpy())
                np.save(save_root + '/{}_GT_fold{}.npy'.format(file[0],i), v_HR.cpu().detach().numpy())
                np.save(save_root + '/{}_res_fold{}.npy'.format(file[0],i), result_val.cpu().detach().numpy())

            ave_RSME = np.mean(RMSE)
            ave_PCC = np.mean(PCC)
            ave_SSIM = np.mean(SSIM)
            ave_PSNR = np.mean(PSNR)

            print(f'\nfold:{i},sample:{sample_list[test_index]},RMSE:{ave_RSME} PCC:{ave_PCC},SSIM:{ave_SSIM} PSNR:{ave_PSNR}')
            loo_PCC.append(ave_PCC)
            loo_rmse.append(ave_RSME)
            loo_SSIM.append(ave_SSIM)
            loo_PSNR.append(ave_PSNR)


    loo_ave_pcc=np.mean(loo_PCC)
    loo_ave_rmse=np.mean(loo_rmse)
    loo_std_pcc=np.std(loo_PCC)
    loo_std_rmse=np.std(loo_rmse)

    loo_ave_SSIM=np.mean(loo_SSIM)
    loo_ave_PSNR=np.mean(loo_PSNR)
    loo_std_SSIM=np.std(loo_SSIM)
    loo_std_PSNR=np.std(loo_PSNR)

    print(f'loo_ave_pcc:{loo_ave_pcc} loo_ave_SSIM:{loo_ave_SSIM}loo_ave_PSNR:{loo_ave_PSNR} loo_ave_rmse:{loo_ave_rmse},loo_std_pcc:{loo_std_pcc} loo_std_rmse:{loo_std_rmse},loo_std_ssim:{loo_std_SSIM} loo_std_PSNR:{loo_std_PSNR}')
    Fold=[i for i in range(len(sample_list))]
    df_save = pd.DataFrame({"Fold":Fold,'sample':sample_list,'PCC': loo_PCC, 'RMSE': loo_rmse,'SSIM': loo_SSIM, 'PSNR': loo_PSNR})

    new_rows = pd.DataFrame({
        'Fold': ['Ave', 'std'],
        'sample': ['Ave', 'std'],
        'PCC': [loo_ave_pcc, loo_std_pcc],
        'RMSE': [loo_ave_rmse, loo_std_rmse],
        'SSIM': [loo_ave_SSIM, loo_std_SSIM],
        'PSNR': [loo_ave_PSNR, loo_std_PSNR]
    })
    new_df = df_save._append(new_rows, ignore_index=True)
    print(new_df)

    save_root_Metrics = dir_root + '/Test/Metrics'
    os.makedirs(save_root_Metrics, exist_ok=True)
    new_df.to_csv(save_root_Metrics+"/Results.csv",index=False)

def test_hyper(args):
    print("\n>> Test model...")
    print('=' * 4+ 'Hyper test information' + '=' * 4)
    args.train_data_root=args.data_root + f'/{args.tissue}/Input'

    if args.test_tissue in ["VisiumHD","Slide-seq"]:
        args.data_root = args.data_root + f'/Simulated/{args.test_tissue}/Input'
    else:
        args.data_root = args.data_root + f'/Independent_test/{args.test_tissue}/Input'

    independenttest_sample_list=np.array(pd.read_csv(args.data_root+f'/{args.test_tissue}_samplelist.csv')['Sample'].tolist())
    train_sample_list=np.array(pd.read_csv(args.train_data_root+f'/{args.tissue}_samplelist.csv')['Sample'].tolist())
    loo=LeaveOneOut()
    loo.get_n_splits(train_sample_list)
    dir_root = f'./result/{args.tissue}/{args.model_name}/'

    for i,(_,_) in enumerate(loo.split(train_sample_list)):
        if i > -1:
            model_save_root=os.path.join(dir_root,'Paras')
            print(f'Fold:{i},sample_list:{independenttest_sample_list}')
            validdata = MyDataset(args, samples=independenttest_sample_list)
            validloader = torch.utils.data.DataLoader(
                dataset=validdata,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8
            )
            model = getattr(module_arch, args.model_name)(args,in_chans=args.in_ch,gene_num=args.gene_num).to(args.device)
            model.load_state_dict(torch.load(model_save_root + f"/finalmodel_fold{i}.pth"), False)
            model.eval()
            index_val=0

            with torch.no_grad():
                if args.tissue=='Slideseq_Mus':
                    base_gene_num = 128
                else:
                    base_gene_num = 512#if tissue==Slideseq_Mus,it is 128,otherwise is 512
                NO=int(args.gene_num / base_gene_num)
                for v, v_data in enumerate(validloader):
                    v_HR = v_data
                    v_HR=v_HR.to(args.device)

                    if (args.tissue=='Slideseq_Mus' and args.gene_num<=128) or (args.tissue!='Slideseq_Mus' and args.gene_num<=512):
                        result_val = model(v_HR)
                        index_val += len(result_val)
                        print(result_val.shape)
                    else:
                        w=v_HR.shape[-1]*2
                        result_val=np.zeros((args.gene_num,w,w))
                        for n in range(NO):
                            # print(f'start index:{n*base_gene_num},end_index:{(n+1)*base_gene_num}')
                            sub_result_val=model(v_HR[:,:,n*base_gene_num:(n+1)*base_gene_num]).squeeze().cpu().detach().numpy()
                            result_val[n*base_gene_num:(n+1)*base_gene_num]=sub_result_val

                    if args.test_tissue in ["VisiumHD", "Slide-seq"]:
                        obj='Simulated'
                    else:
                        obj = 'independent_test'
                    save_root = dir_root + f'/Hyper_{obj}_{args.test_tissue}_{args.gene_num}/Output'
                    os.makedirs(save_root, exist_ok=True)
                    np.save(save_root + '/{}_HR_fold{}.npy'.format(independenttest_sample_list[v],i), v_HR.cpu().detach().numpy())
                    np.save(save_root + '/{}_res_fold{}.npy'.format(independenttest_sample_list[v],i), result_val)
