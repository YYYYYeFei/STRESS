"""
Data:20250506
Author:Fei Ye
"""
import argparse
import mode as mode_arch
import torch

def main():
    parser = argparse.ArgumentParser()
    # 超参数
    parser.add_argument("--batch_size", default=1, type=int,help="Total batch size for training.")
    parser.add_argument("--in_ch", default=4, type=int,help="channels for the input of model")
    parser.add_argument("--learning_rate", default=1e-4, type=float,help="The initial learning rate for SGD.")
    parser.add_argument("--num_epoch", default=500, type=int,help="Total number of training epochs to perform.")
    parser.add_argument("--patience", default=5, type=int,help="early stopping.",required=False)
    parser.add_argument("--optimizer", default="Adam", type=str,help="the base deeplearning optimizer",required=False)
    parser.add_argument("--model_name", default="STRESS", type=str,help="model name",required=False)
    parser.add_argument("--GPU_id", default=0, type=str,help="GPU id.", required=False)
    parser.add_argument("--mode", default="train", type=str, help="functions of STRESS")
    parser.add_argument("--test_tissue", default=None, type=str,help="Testing dataname")
    parser.add_argument("--tissue", default=None, type=str,help="Training dataname")
    parser.add_argument("--independent_tissue", default=None, type=str, help="independent dataname")
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--sampling',default="all", dest='sampling', type=str)
    parser.add_argument('--gene_num', default=1024, dest='gene_num', type=int)
    parser.add_argument('--start_fold', default=-1, dest='start_fold', type=int)
    parser.add_argument('--data_root', default='/data1/yefei/code/ST/STRESS/data', type=str)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.GPU_id)) if torch.cuda.is_available() else torch.device('cpu')


    print("\n"+"*"+"=" * 20 + "超参数信息" + "=" * 20+"*")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print("\nStart {}.....".format(args.mode))
    getattr(mode_arch, args.mode)(args)


if __name__ == '__main__':
    main()