import argparse
import collections
import itertools
import torch
from torch.optim.lr_scheduler import StepLR
import random
import os

from model import nbimvc
from get_indicator_matrix_A import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config


def main(MR):
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['dataset'] = dataset
    print("Data set: " + config['dataset'])
    print("MissRate: " + str(MR))
    config['print_num'] = config['training']['epoch'] / 10     # print_num
    logger = get_logger()

    # Load data
    seed = config['training']['seed']
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]

    for missingrate in MR:
        lambda_1_values = [0.01, 0.1, 1, 10, 100]
        lambda_2_values = [0.01, 0.1, 1, 10, 100]
        lambda_3_values = [0.01, 0.1, 1, 10, 100]
        for lambda_1 in lambda_1_values:
            for lambda_2 in lambda_2_values:
                for lambda_3 in lambda_3_values:
                    config['training']['missing_rate'] = missingrate
                    # 保存结果
                    best_scores_kmeans = [0, 0, 0]
                    best_scores_q = [0, 0, 0, 0]
                    print('--------------------Missing rate = ' + str(missingrate) + '--------------------')
                    for data_seed in range(1, args.test_time + 1):
                        # get the mask
                        np.random.seed(data_seed)
                        mask = get_mask(config['training']['view'], x1_train_raw.shape[0], config['training']['missing_rate'])
                        # mask the data
                        X_train_list = [X_list[i] * mask[:, i][:, np.newaxis] for i in range(len(X_list))]
                        X_train_list = [torch.from_numpy(X_train_list[i]).float().to(device) for i in range(len(X_train_list))]

                        mask = torch.from_numpy(mask).long().to(device)

                        # Set random seeds for model initialization
                        np.random.seed(seed)
                        random.seed(seed)
                        os.environ['PYTHONHASHSEED'] = str(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = True

                        # Build the model
                        NBIMVC = nbimvc(config, device)
                        optimizer = torch.optim.Adam(
                            itertools.chain(NBIMVC.parameters()),
                                        lr=config['training']['lr'],
                                        weight_decay=0.001,)
                        # Pre-training
                        epoch = 1
                        while epoch <= args.pre_epochs:
                            NBIMVC.train_pre(config, logger, X_train_list, Y_list, mask, optimizer, epoch, device)
                            epoch += 1

                        while epoch <= args.pre_epochs + args.con_epochs:
                            feature_all = NBIMVC.train_con(config, logger, X_train_list, Y_list, mask, optimizer, epoch, device, lambda_1, lambda_2, lambda_3)
                            # scheduler.step()  # 更新学习率
                            for v in range(config['training']['view']):
                                predicted = NBIMVC.kmeans_layer[v].fit_predict(feature_all[v])
                                cluster_centers = torch.tensor(
                                    NBIMVC.kmeans_layer[v].cluster_centers_, dtype=torch.float, requires_grad=True
                                )  # [10, 512]
                                with torch.no_grad():
                                    NBIMVC.pseudos[v].cluster_centers.copy_(cluster_centers)
                            if epoch % 1 == 0:
                                with torch.no_grad():
                                    scores_kmeans, scores_q,_,_ = NBIMVC.valid(config, logger, mask, X_train_list, Y_list, device)
                                    if scores_kmeans[0] > best_scores_kmeans[0]:
                                        best_scores_kmeans = scores_kmeans
                                    if scores_q[0] > best_scores_q[0]:
                                        best_scores_q = scores_q
                            epoch += 1


                        # Testing
                        print('Lambda_1: = {:.3f} Lambda_2: = {:.3f} Lambda_3: = {:.3f}'.format(lambda_1, lambda_2, lambda_3))
                        print('---------------------------Result of {}----------------------------'.format(dataset))
                        print('------------------------- MissRate = {:.2f}--------------------'.format(missingrate))
                        print('K-meansClustering: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(best_scores_kmeans[0], best_scores_kmeans[1], best_scores_kmeans[2]))
                        print('Clustering: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR = {:.4f}'.format(best_scores_q[0], best_scores_q[1], best_scores_q[2], best_scores_q[3]))
                        print('------------------------Training over------------------------')

if __name__ == '__main__':
    dataset = {0: "MNIST-USPS",
               1: "Caltech101-20",
               2: "RGB-D",
               3: "Scene-15",
               4: "NoisyMNIST",
               5: "BDGP",
               6: "HandWritten",
               8: "LandUse-21",}
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=str(6), help='dataset id')  # data index
    parser.add_argument('--test_time', type=int, default=str(1), help='number of test times')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    parser.add_argument('--pre_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--con_epochs', type=int, default=120, help='number of training epochs')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    MisingRate = [0.1]
    main(MR=MisingRate)

    # nohup python run_iter.py > output_lambda_handwritten_0.1.log 2>&1 &
