import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import numpy as np
from torch.nn import Parameter
from typing import Optional
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors

import evaluation
from util import next_batch
from instance_loss import InstanceLoss_neg
from labelcon_loss import Loss
from Metric import *
from sklearn.cluster import KMeans
    
class DDClustering(nn.Module):
    def __init__(self, inputDim, n_cluster):
        super(DDClustering, self).__init__()
        hidden_layers = [nn.Linear(inputDim, 256), nn.ReLU()]
        hidden_layers.append(nn.BatchNorm1d(num_features=256))
        self.hidden = nn.Sequential(*hidden_layers)
        self.withoutSoft = nn.Linear(256, n_cluster)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.hidden(x)
        withoutSoftMax = self.withoutSoft(hidden)
        output = self.output(withoutSoftMax)
        return output
    
class ClusterAssignment(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True), self.cluster_centers


def complete_missing_view(view1_common, view1_only, k=3):
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(view1_common.cpu().detach().numpy())

    nearest_indices = []
    for sample in view1_only:
        _, indices = knn_model.kneighbors(sample.unsqueeze(0).cpu().detach().numpy())
        nearest_indices.append(indices.squeeze())

    return torch.tensor(np.array(nearest_indices)).long()


def cosine_similarity(features1, features2):
    features1_normalized = F.normalize(features1, p=2, dim=1)
    features2_normalized = F.normalize(features2, p=2, dim=1)
    similarity_matrix = torch.matmul(features1_normalized, features2_normalized.t())
    return similarity_matrix

def construct_positive_pairs(features_view1, features_view2, Ins_con_loss_fn):
    positive_pairs = []
    similarity_matrix = cosine_similarity(features_view1, features_view2)
    ins_loss_all = 0.0
    for i in range(similarity_matrix.size(0)):
        _, most_similar_index = similarity_matrix[i].max(dim=0)
        positive_pairs.append((i, most_similar_index.item()))
        
        ins_loss_all += Ins_con_loss_fn(features_view1[i], features_view2[most_similar_index], 1)
        
    return positive_pairs, ins_loss_all

class Autoencoder(nn.Module):
    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):

        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers = decoder_layers[:-1]
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):

        latent = self._encoder(x)
        return latent

    def decoder(self, latent):

        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):

        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class nbimvc(nn.Module):

    def __init__(self, config, device):
        super(nbimvc, self).__init__()
        self._config = config
        self.view = config['training']['view']

        self._latent_dim = config['Autoencoder']['arch1'][-1]

        self.class_dim = config['training']['n_class']

        self.autoencoders = nn.ModuleList()  # 用于存储多个编码器
        for i in range(self.view):
            arch = config['Autoencoder'][f'arch{i}']
            activations = config['Autoencoder'][f'activations{i}']
            batchnorm = config['Autoencoder']['batchnorm']
            self.autoencoders.append(Autoencoder(arch, activations, batchnorm).to(device))

        self.cluster = DDClustering(self._latent_dim, self.class_dim).to(device)
        self.kmeans_layer = []
        self.pseudos = []
        for v in range(self.view):
            self.kmeans_layer.append(KMeans(n_clusters=config['training']['n_class'], n_init=20))
            self.pseudos.append(ClusterAssignment(self.class_dim, self._latent_dim).to(device))
        self.ins_neg = InstanceLoss_neg(temperature=config['training']['temperature'])
        self.label_con = Loss(config['training']['batch_size'], config['training']['n_class'], 1, 1, device)


    def train_pre(self, config, logger, X_train_list, Y_list, mask, optimizer, epoch, device):
        for autoencoder in self.autoencoders:
            autoencoder.train()
        self.cluster.train()
        total_loss = 0
        flag = (mask == 1).int()
        for v1 in range(self.view-1):
            for v2 in range (v1+1, self.view):
                X1, X2, X3, X4 = shuffle(X_train_list[v1], X_train_list[v2], flag[:, v1], flag[:, v2])
                for batch_x1, batch_x2, x1_index, x2_index, batch_No in next_batch(X1, X2, X3, X4, config['training']['batch_size']):
                    if len(batch_x1) == 1:
                        continue
                    z_1 = self.autoencoders[v1].encoder(batch_x1[x1_index == 1])
                    z_2 = self.autoencoders[v2].encoder(batch_x2[x2_index == 1])
                    recon1 = F.mse_loss(self.autoencoders[v1].decoder(z_1), batch_x1[x1_index == 1])
                    recon2 = F.mse_loss(self.autoencoders[v2].decoder(z_2), batch_x2[x2_index == 1])

                    rec_loss = recon1 + recon2
                    optimizer.zero_grad()
                    rec_loss.backward()
                    optimizer.step()
                    total_loss += rec_loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss / (X_train_list[0].shape[0])))

    def train_con(self, config, logger, X_train_list, Y_list, mask, optimizer, epoch, device, lambda_1, lambda_2, lambda_3):
        for autoencoder in self.autoencoders:
            autoencoder.train()
        for pseudo in self.pseudos:
            pseudo.train()
        self.cluster.train()
        Z_all = []
        feature_all = []
        total_loss = 0
        flag = (mask == 1).int()
        for v in range(self.view):
            Z_all.append([])
            feature_all.append([])
        for v1 in range(self.view-1):
            for v2 in range(v1+1, self.view):
                X1, X2, X3, X4 = shuffle(X_train_list[v1], X_train_list[v2], flag[:, v1], flag[:, v2])
                for batch_x1, batch_x2, x1_index, x2_index, batch_indices in next_batch(X1, X2, X3, X4, config['training']['batch_size']):
                    if len(batch_x1) == 1:
                        continue
                    index_both = x1_index + x2_index == 2                      # 视角 1 和视角 2 都存在
                    index_peculiar1 = (x1_index + x1_index + x2_index == 2)    # 只有视角 1
                    index_peculiar2 = (x1_index + x2_index + x2_index == 2)    # 只有视角 2
                    z_1 = self.autoencoders[v1].encoder(batch_x1[x1_index == 1])
                    z_2 = self.autoencoders[v2].encoder(batch_x2[x2_index == 1])

                    recon1 = F.mse_loss(self.autoencoders[v1].decoder(z_1), batch_x1[x1_index == 1])
                    recon2 = F.mse_loss(self.autoencoders[v2].decoder(z_2), batch_x2[x2_index == 1])
                    rec_loss = (recon1 + recon2)

                    z_view1_both = self.autoencoders[v1].encoder(batch_x1[index_both])
                    z_view2_both = self.autoencoders[v2].encoder(batch_x2[index_both])

                    if len(batch_x2[index_peculiar2]) % config['training']['batch_size'] == 1:
                        continue
                    z_view2_peculiar = self.autoencoders[v2].encoder(batch_x2[index_peculiar2])

                    if len(batch_x1[index_peculiar1]) % config['training']['batch_size'] == 1:
                        continue
                    z_view1_peculiar = self.autoencoders[v1].encoder(batch_x1[index_peculiar1])

                    # 临近样本补全
                    nearest_indices_1 = complete_missing_view(z_view1_both, z_view1_peculiar, 5)
                    z_view2_missed = torch.mean(z_view2_both[nearest_indices_1], dim=1).detach()

                    nearest_indices_2 = complete_missing_view(z_view2_both, z_view2_peculiar, 5)
                    z_view1_missed = torch.mean(z_view1_both[nearest_indices_2], dim=1).detach()

                    view1 = torch.cat([z_view1_both, z_view1_peculiar, z_view1_missed], dim=0)
                    view2 = torch.cat([z_view2_both, z_view2_missed, z_view2_peculiar], dim=0)
                    Z_all[v1].append(view1)
                    Z_all[v2].append(view2)

                    q1 = self.cluster(view1)
                    q2 = self.cluster(view2)
                    pseudo_q1, center1 = self.pseudos[v1](view1)
                    pseudo_q2, center2 = self.pseudos[v2](view2)

                    cor_loss = self.ins_neg(z_view1_both, z_view2_both)
                    label_loss_con = self.label_con.forward_label_all(q1, q2, pseudo_q1, pseudo_q2)

                    loss =  rec_loss * lambda_1 + label_loss_con * lambda_2 + lambda_3 * cor_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss / len(X1)))

        for v in range(self.view):
            feature_all[v] = torch.cat(Z_all[v], dim=0).detach().cpu().numpy()
        
        return feature_all


    def valid(self, config, logger, mask, X_train_list, Y_list, device):
        with torch.no_grad():
            for autoencoder in self.autoencoders:
                autoencoder.eval()
            for pseudo in self.pseudos:
                pseudo.eval()
            self.cluster.eval()

            filled_views = torch.zeros(self.view, len(X_train_list[0]), 128).to(device)
            Y_views = torch.zeros(len(X_train_list[0]), dtype=torch.int32).to(device)
            view_counts = torch.zeros(self.view, len(X_train_list[0])).to(device)

            Y_list = torch.tensor(np.array(Y_list)).int().to(device).squeeze(dim=0).unsqueeze(dim=1).detach()
            for v in range(self.view-1):
                flag_filled = (mask[:, v] + mask[:, self.view - 1]) >= 1  # 视角 1 或 视角 2 中至少一个存在
                flag = mask[:, v] + mask[:, self.view-1] == 2       # 视角 1 和视角 2 都存在
                view1_only_idx_eval = (mask[:, v] == 1) & (mask[:, self.view - 1] == 0)     # 只有视角 1
                view2_only_idx_eval = (mask[:, self.view - 1] == 1) & (mask[:, v] == 0)     # 只有视角 2

                common_view1 = X_train_list[v][flag]
                common_view1 = self.autoencoders[v].encoder(common_view1)
                common_view2 = X_train_list[self.view-1][flag]
                common_view2 = self.autoencoders[self.view-1].encoder(common_view2)
                y_common = Y_list[flag]

                only_view1_exist = X_train_list[v][view1_only_idx_eval]
                z_view1_exist = self.autoencoders[v].encoder(only_view1_exist)
                y_view1_exist = Y_list[view1_only_idx_eval]

                only_view2_exist = X_train_list[self.view-1][view2_only_idx_eval]
                z_view2_exist = self.autoencoders[self.view-1].encoder(only_view2_exist)
                y_view2_exist = Y_list[view2_only_idx_eval]

                # 临近样本补全
                nearest_indices_1 = complete_missing_view(common_view1, z_view1_exist, k=3)
                z_view2_missed = torch.mean(common_view2[nearest_indices_1], dim=1)
                nearest_indices_2 = complete_missing_view(common_view2, z_view2_exist, k=3)
                z_view1_missed = torch.mean(common_view1[nearest_indices_2], dim=1)

                filled_views[v, flag] = common_view1; filled_views[v, view1_only_idx_eval] = z_view1_exist; filled_views[v, view2_only_idx_eval] = z_view1_missed
                view_counts[v, flag_filled] += 1
                filled_views[self.view-1, flag] = common_view2; filled_views[self.view-1, view1_only_idx_eval] = z_view2_missed; filled_views[self.view-1, view2_only_idx_eval] = z_view2_exist
                view_counts[self.view - 1, flag_filled] += 1

                Y_views[flag] = Y_list[flag].squeeze(dim=1)
                Y_views[view1_only_idx_eval] = Y_list[view1_only_idx_eval].squeeze(dim=1)
                Y_views[view2_only_idx_eval] = Y_list[view2_only_idx_eval].squeeze(dim=1)

            Y_list_last = Y_views
            filled_views_concat = filled_views.permute(1, 0, 2).reshape(filled_views.shape[1], -1)
            view_both = (filled_views_concat / view_counts.sum(0).clamp(min=1).unsqueeze(1))

            cluster_outputs = []
            for v in range(self.view):
                cluster_output = self.cluster(filled_views[v])
                cluster_outputs.append(cluster_output)

            Q = torch.stack(cluster_outputs, dim=0).mean(dim=0)

            soft_vector = Q.cpu().detach().numpy()
            total_pred = np.argmax(np.array(soft_vector), axis=1)
            nmi, ari, acc, pur = evaluate(Y_list_last.cpu().numpy(), total_pred)

            # kmeans进行测试
            latent_fusion = view_both.cpu().detach().numpy()
            scores, _ = evaluation.clustering([latent_fusion], Y_list_last.cpu().numpy())

        # print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
        # print('K-means: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(scores['kmeans']['ACC'], scores['kmeans']['NMI'], scores['kmeans']['ARI']))
        for autoencoder in self.autoencoders:
            autoencoder.train()
        for pseudo in self.pseudos:
            pseudo.train()
        self.cluster.train()
        return [scores['kmeans']['ACC'], scores['kmeans']['NMI'], scores['kmeans']['ARI']], [acc, nmi, ari, pur], latent_fusion, Y_list_last.cpu().numpy()
