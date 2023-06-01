import torch
from torch import nn
import torch.nn.functional as F

import torch
class ContrastiveLoss3(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.25):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵
        self.negatives_mask = torch.zeros((2*batch_size, 2*batch_size)).to('cuda')
        self.negatives_mask[0:batch_size, batch_size:2*batch_size]=1
        self.negatives_mask[batch_size:2 * batch_size, 0:batch_size] = 1
    def forward(self, emb_i, emb_j,batch_size=None):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        # z_i = emb_i  # (bs, dim)  --->  (bs, dim)
        # z_j = emb_j  # (bs, dim)  --->  (bs, dim)
        if batch_size !=None:
            self.batch_size = batch_size
            self.register_buffer("negatives_mask", (
                ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to('cuda')).float())
        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        # similarity_matrix = F.cosine_similarity(z_i, z_j,
        #                                         dim=1)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs


        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.75, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")

            self.temperature = torch.tensor(self.temperature).cuda()
            numerator = torch.exp(sim_i_j / self.temperature)
            zero_= torch.tensor(0.0).cuda()
            one_for_not_i = torch.ones((2 * self.batch_size,)).to("cuda:0").scatter_(0, torch.tensor([i]).cuda(), zero_)
            if self.verbose: print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss

import torch
from torch import nn


def distanceL2(h, t):
    s = h - t
    sum = torch.square(s).sum(-1)
    return sum

def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def l2_sim(im, s):
    b_im = im.shape[0]
    b_s = s.shape[0]
    return distanceL2(im.unsqueeze(0).repeat(b_s,1,1),s.unsqueeze(1).repeat(1,b_im,1)).transpose(0,1)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.5, measure=False, max_violation=False):
        # max_violation 是否用最难样本
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.measure = measure
        if measure == 'l2':
            self.sim = l2_sim
            # self.margin = -self.margin
        if measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s, matrix):

        matrix = torch.tensor(matrix).cuda()
        # compute image-sentence score matrix
        #im,s维度相同，默认将除了配对的都视为负样本
        im = F.normalize(im, dim=1)
        s = F.normalize(s, dim=1)

        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column

        if self.measure == 'l2':
            # h+r, t-
            cost_s = (self.margin + d1  - scores).clamp(min=0)
            # compare every diagonal score to scores in its row
            # (h+r)-, t
            cost_im = (self.margin + d2  - scores).clamp(min=0)
        else:
            # h+r, t-; 0.3+0.2 -0.8<0; 0.7+0.2-0.6>0
            cost_s = (self.margin + scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # (h+r)-, t
            cost_im = (self.margin + scores - d2).clamp(min=0)

        # cost_s = (self.margin + scores - d1).clamp(min=0)
        # # compare every diagonal score to scores in its row
        # # (h+r)-, t
        # cost_im = (self.margin + scores - d2).clamp(min=0)

        # # clear diagonals
        # mask = torch.eye(scores.size(0)) > .5
        # I = mask
        # if torch.cuda.is_available():
        #     I = I.cuda()
        # cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # another mask method
        mask1 = scores.eq(d1).cuda()
        mask2 = mask1.t()
        mask3 = matrix.eq(1).cuda()

        cost_s = cost_s.masked_fill_(mask1, 0)
        cost_im = cost_im.masked_fill_(mask2, 0)

        cost_s = cost_s.masked_fill_(mask3, 0)
        cost_im = cost_im.masked_fill_(mask3, 0)


        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return (cost_s.sum() + cost_im.sum())/(cost_s.shape[0]*cost_s.shape[1]-mask3.sum()-cost_s.shape[0])
        # return cost_s.sum() + cost_im.sum()
