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
    config['print_num'] = config['training']['epoch'] / 10
    logger = get_logger()
    args.pre_epochs = config['training']['pre_epochs']
    args.con_epochs = config['training']['con_epochs']

    # Load data
    seed = config['training']['seed']
    X_list, Y_list = load_data(config)

    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]

    for missingrate in MR:
        accumulated_metrics = collections.defaultdict(list)
        config['training']['missing_rate'] = missingrate
        # 保存结果00
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
                            weight_decay=0.02,
                        )  # weight_decay=0.01

            # Pre-training
            epoch = 1
            while epoch <= args.pre_epochs:
                NBIMVC.train_pre(config, logger, X_train_list, Y_list, mask, optimizer, epoch, device)
                epoch += 1

            while epoch <= args.pre_epochs + args.con_epochs:
                feature_all = NBIMVC.train_con(config, logger, X_train_list, Y_list, mask, optimizer, epoch, device, config['training']['lambda1'], 
                                              config['training']['lambda2'], config['training']['lambda3'])
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
                            best_model_state = NBIMVC.state_dict()
                        if scores_q[0] >= best_scores_q[0]:
                            best_scores_q = scores_q
                epoch += 1

            # Testing
            print('---------------------------Result of {}----------------------------'.format(dataset))
            print('------------------------- MissRate = {:.2f}--------------------'.format(missingrate))
            print('K-meansClustering: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(best_scores_kmeans[0], best_scores_kmeans[1], best_scores_kmeans[2]))
            print('Clustering: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR = {:.4f}'.format(best_scores_q[0], best_scores_q[1], best_scores_q[2], best_scores_q[3]))
            print('------------------------Training over------------------------')
        if best_model_state is not None:
            torch.save(best_model_state, f"ckpt/best_model_{dataset}_missingrate_{missingrate}.pt")


if __name__ == '__main__':
    dataset = {0: "MNIST-USPS",
               1: "Scene-15",
               2: "BDGP",
               3: "HandWritten",
               4: "Caltech101-7",
               5: "LandUse-21",}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=str(1), help='dataset id')  # data index
    parser.add_argument('--test_time', type=int, default=str(1), help='number of test times')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    parser.add_argument('--pre_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--con_epochs', type=int, default=50, help='number of training epochs')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    MisingRate = [0.7]  # [0.1, 0.3, 0.5, 0.7]
    main(MR=MisingRate)

# nohup python run.py > output_MNISTUSPS_0.7.log 2>&1 &
# nohup python run.py > output_Scene-15_0.7.log 2>&1 &
# nohup python run.py > output_RGBD_10_0.5.log 2>&1 &
# nohup python run.py > output_LandUse_21.log 2>&1 &