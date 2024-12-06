import torch
import torch.nn as nn
import torch.nn.functional as F

    
class InstanceLoss_neg(nn.Module):
    def __init__(self, batch_size=256, temperature=0.5, neg_samples_per_pos=1):
        super(InstanceLoss_neg, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.neg_samples_per_pos = neg_samples_per_pos
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # Hard Negative Mining
        sim = sim.clone().detach()
        sim.fill_diagonal_(float('-inf'))
        topk_values, _ = sim.topk(self.neg_samples_per_pos, dim=1)
        topk_values = topk_values[:, -1]
        negative_samples = topk_values.reshape(N, 1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
        
class Instance_Align_Loss(nn.Module):
    def __init__(self):
        super(Instance_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss2 = mse(gt, P)

        return Loss2