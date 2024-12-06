import torch
import torch.nn as nn
import torch.nn.functional as F
import math
eps=1e-8


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy
    
    
    def forward_label_all(self, q_i, q_j, pseudo_i, pseudo_j):
        loss_function = torch.nn.KLDivLoss(reduction='batchmean')
        loss_list = []
        labels = [q_i, q_j]
        pseudos = [pseudo_i, pseudo_j]
        for v in range(2):
            for w in range(v+1, 2):
                loss_list.append(
                    (loss_function(pseudos[v].log(), target_distribution(pseudos[w]).detach()) /
                     pseudos[v].shape[0]))
                loss_list.append(self.forward(labels[v], labels[w]))
            loss_list.append((loss_function(pseudos[v].log(), target_distribution(pseudos[v]).detach()) /
                              pseudos[v].shape[0]))
            loss_list.append(
                (loss_function(labels[v].log(), target_distribution(pseudos[v]).detach()) /
                 pseudos[v].shape[0]))

        return sum(loss_list)
    

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()