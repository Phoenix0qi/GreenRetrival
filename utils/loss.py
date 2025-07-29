"""
Loss functions
"""

import torch
import torch.nn as nn


def chamfer_loss(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    xy = torch.bmm(x, y.transpose(2, 1))
    dtype = torch.cuda.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)

    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xy.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(xy)
    P = (rx.transpose(2, 1) + ry - 2 * xy)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.mean(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.mean(mins)

    return loss_1 + loss_2

def chamfer_loss_x(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    xy = torch.bmm(x, y.transpose(2, 1))
    dtype = torch.cuda.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)

    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xy.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(xy)
    P = (rx.transpose(2, 1) + ry - 2 * xy)  # [batch_size, num_points_x, num_points_y]
    mins, _ = torch.min(P, 1)
    loss_1 = torch.mean(mins)

    return loss_1


def calculate_angle_loss(theta_vector_test, theta_vector_gt):
    """计算每个测试向量与所有 GT 向量之间的最小夹角"""
    # theta_vector_test: [batch, 128, 3]
    # theta_vector_gt: [batch, 4, 3]

    loss = 0.0

    n_points = theta_vector_test.shape[1]

    norm_test = torch.norm(theta_vector_test, dim=-1, keepdim=True)  # [batch, 128, 1]
    norm_gt = torch.norm(theta_vector_gt, dim=-1, keepdim=True)      # [batch, 4, 1]

    # 计算余弦相似度
    cos_similarity = torch.einsum('ijk,ilk->ijl', theta_vector_test / norm_test, theta_vector_gt / norm_gt)


    # 找到每个测试向量对应的最小角度
    min_cos_sim, _ = torch.max(cos_similarity, dim=2)  # [batch, 128]

    # 计算小于等于10的角度的损失
    loss = torch.sum(1 - min_cos_sim)  # 小于等于10的角度的总和

    return loss/n_points


def calculate_angle_loss_Ez_theta(theta_vector_test, theta_vector_gt, scatter_Ez_pred, scatter_Ez_gt):
    """计算每个测试向量与所有 GT 向量之间的最小夹角"""
    # theta_vector_test: [batch, 128, 3]
    # theta_vector_gt: [batch, 9, 3]

    loss = 0.0
    epsilon = 1e-8

    n_points = theta_vector_test.shape[1]

    norm_test = torch.norm(theta_vector_test, dim=-1, keepdim=True) + epsilon  # [batch, 128, 1]
    norm_gt = torch.norm(theta_vector_gt, dim=-1, keepdim=True) + epsilon      # [batch, 9, 1]

    # 计算余弦相似度
    cos_similarity = torch.einsum('ijk,ilk->ijl', theta_vector_test / norm_test, theta_vector_gt / norm_gt) # [batch, n_scatter, 9]

    # 找到每个测试向量对应的最小角度
    min_cos_sim, index = torch.max(cos_similarity, dim=2)  # [batch, 128]

    # 如果 min_cos_sim 小于20度，把对应的scatter_Ez_pred
    threshold = torch.cos(torch.tensor(30 * (3.141592 / 180)))
    mask = min_cos_sim >= threshold
    # scatter_sum_pred = torch.zeros_like(scatter_Ez_gt).clone()
    scatter_sum_pred = torch.zeros_like(scatter_Ez_gt).expand(-1,-1,2).clone()
    # 使用掩码和索引将符合条件的 scatter_Ez_pred 累加到 scatter_sum_pred 中
    mask = mask.float().unsqueeze(dim=-1)
    scatter_sum_pred.scatter_add_(1, index.unsqueeze(dim=-1).expand(-1,-1,2), scatter_Ez_pred * mask + epsilon)  # [bz, 9, 2]

    # Ez_theta_loss_mse = nn.functional.mse_loss(scatter_sum_pred, scatter_Ez_gt)

    # 计算小于等于10的角度的损失
    loss = torch.sum(1 - min_cos_sim)  # 小于等于10的角度的总和

    return loss/n_points, scatter_sum_pred, scatter_Ez_gt

class MSE_Ez_conj___chamfer_loss___orient_cos_loss(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']
        self.alpha_orient_cos_loss = h['alpha_orient_cos_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]  # scatters_gt_xyz: [batch, 10, 9]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], -1*labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], -1*labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])

        rloc = labels[2][1]
        theta_vector_gt = labels[2][0][:, 0:9, 3:6] - rloc.unsqueeze(1)
        theta_vector_test = outputs[1] - rloc.unsqueeze(1)
        orient_cos_loss = self.orient_cos_loss_fn(theta_vector_test, theta_vector_gt)

        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss + orient_cos_loss * self.alpha_orient_cos_loss

        return [loss, Ez_loss_conj, loss_rp, orient_cos_loss]

class MSE_Ez_reverse___chamfer_loss___orient_cos_loss(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']
        self.alpha_orient_cos_loss = h['alpha_orient_cos_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]  # scatters_gt_xyz: [batch, 10, 9]
        # 在__call__方法中执行损失计算
        # Ez_loss_conj = self.MSE_loss_fn(outputs[0], labels[0])
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])

        rloc = labels[2][1]
        theta_vector_gt = labels[2][0][:, 0:9, 3:6] - rloc.unsqueeze(1)
        theta_vector_test = outputs[1] - rloc.unsqueeze(1)
        orient_cos_loss = self.orient_cos_loss_fn(theta_vector_test, theta_vector_gt)

        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss + orient_cos_loss * self.alpha_orient_cos_loss

        return [loss, Ez_loss_conj, loss_rp, orient_cos_loss]



class MSE_Ez_reverse___chamfer_loss_x___orient_cos_loss(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']
        self.alpha_orient_cos_loss = h['alpha_orient_cos_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]  # scatters_gt_xyz: [batch, 10, 9]
        # 在__call__方法中执行损失计算
        # Ez_loss_conj = self.MSE_loss_fn(outputs[0], labels[0])
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])

        rloc = labels[2][1]
        theta_vector_gt = labels[2][0][:, 0:9, 3:6] - rloc.unsqueeze(1)
        theta_vector_test = outputs[1] - rloc.unsqueeze(1)
        orient_cos_loss = self.orient_cos_loss_fn(theta_vector_test, theta_vector_gt)

        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss + orient_cos_loss * self.alpha_orient_cos_loss

        return [loss, Ez_loss_conj, loss_rp, orient_cos_loss]

class MSE_Ez_reverse___chamfer_loss_x___orient_cos_loss_Ez_theta(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss_Ez_theta
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']
        self.alpha_orient_cos_loss = h['alpha_orient_cos_loss']
        self.alpha_orient_Ez_theta_loss = h['alpha_orient_Ez_theta_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz] if Ez theta:  [Ez_ReIm, [scatters_xyz, scatters_Ez]]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]  # scatters_gt_xyz: [batch, 10, 9]
        # 在__call__方法中执行损失计算
        # Ez_loss_conj = self.MSE_loss_fn(outputs[0], labels[0])
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1][0], labels[1])

        rloc = labels[2][1]
        theta_vector_gt = labels[2][0][:, 0:9, 3:6] - rloc.unsqueeze(1)
        theta_vector_test = outputs[1][0] - rloc.unsqueeze(1)
        # orient_cos_loss, scatter_sum_pred, scatter_Ez_gt = self.orient_cos_loss_fn(theta_vector_test, theta_vector_gt, outputs[1][1], labels[2][0][:, 0:9, 6:8])
        orient_cos_loss, scatter_sum_pred, scatter_Ez_gt = self.orient_cos_loss_fn(theta_vector_test, theta_vector_gt, outputs[1][1], labels[2][0][:, 0:9, 6:7])
        # Ez_theta_loss_mse = nn.functional.mse_loss(scatter_sum_pred, scatter_Ez_gt)
        pred_abs = (scatter_sum_pred[..., 0]**2 + scatter_sum_pred[..., 1]**2) ** 0.5
        pred_abs_ratio = pred_abs / pred_abs.sum()
        gt_abs = 10 ** (scatter_Ez_gt / 20).squeeze(-1)
        gt_abs_ratio = gt_abs / gt_abs.sum()
        Ez_theta_loss_mse = nn.functional.mse_loss(pred_abs_ratio, gt_abs_ratio)

        # Ez_theta_loss_mse = nn.functional.mse_loss((scatter_sum_pred[..., 0]**2 + scatter_sum_pred[..., 1]**2) ** 0.5, (14.1421356 * 10 ** ((scatter_Ez_gt + 60.1569499)/10)).squeeze(-1))

        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss + orient_cos_loss * self.alpha_orient_cos_loss + Ez_theta_loss_mse * self.alpha_orient_Ez_theta_loss

        return [loss, Ez_loss_conj, loss_rp, orient_cos_loss]



class MSE_Ez___chamfer_loss___orient_cos_loss(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']
        self.alpha_orient_cos_loss = h['alpha_orient_cos_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]  # scatters_gt_xyz: [batch, 10, 9]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = self.MSE_loss_fn(outputs[0], labels[0])
        # Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])

        rloc = labels[2][1]
        theta_vector_gt = labels[2][0][:, 0:9, 3:6] - rloc.unsqueeze(1)
        theta_vector_test = outputs[1] - rloc.unsqueeze(1)
        orient_cos_loss = self.orient_cos_loss_fn(theta_vector_test, theta_vector_gt)

        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss + orient_cos_loss * self.alpha_orient_cos_loss

        return [loss, Ez_loss_conj, loss_rp, orient_cos_loss]


class MSE_Ez___chamfer_loss_x(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,0]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,1]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss

        return [loss, Ez_loss_conj, loss_rp, -1]

class MSE_Ez_conj___chamfer_loss_x(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], -1 * labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], -1 * labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss

        return [loss, Ez_loss_conj, loss_rp, -1]


class MSE_Ez_j___chamfer_loss_x(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        # Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], -1 * labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], -1 * labels[0][:,0]))
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], -1*labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss

        return [loss, Ez_loss_conj, loss_rp, -1]

class MSE_Ez(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        # self.chamfer_loss_fn = chamfer_loss_x
        # self.orient_cos_loss_fn = calculate_angle_loss
        # self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,0]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,1]))
        # loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj # + loss_rp * self.alpha_chamfer_loss


        # # # NMSE
        # NMSE = (((outputs[0][:,0] - labels[0][:,0]) ** 2 + (outputs[0][:,1] - labels[0][:,1]) ** 2) / (labels[0][:,0] ** 2 + labels[0][:,1] ** 2 + 1e-8)).mean()
        # loss = NMSE

        return [loss, Ez_loss_conj, -1, -1]

class MSE_Ez_fu(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        # self.chamfer_loss_fn = chamfer_loss_x
        # self.orient_cos_loss_fn = calculate_angle_loss
        # self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], -1 * labels[0][:,0]) + self.MSE_loss_fn(outputs[0][:,1], -1 * labels[0][:,1]))
        # loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj # + loss_rp * self.alpha_chamfer_loss

        # # # NMSE fu
        # NMSE = (((outputs[0][:,0] + labels[0][:,0]) ** 2 + (outputs[0][:,1] + labels[0][:,1]) ** 2) / (labels[0][:,0] ** 2 + labels[0][:,1] ** 2 + 1e-8)).mean()
        # loss = NMSE

        return [loss, Ez_loss_conj, -1, -1]


class MSE_Ez_reverse___chamfer_loss_x(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss

        return [loss, Ez_loss_conj, loss_rp, -1]

class MSE_Ez_reverse___chamfer_loss_x_ckai(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj * 40 # + loss_rp * self.alpha_chamfer_loss

        return [loss, Ez_loss_conj, loss_rp, -1]


class MSE_Ez_reverse_ReIm___chamfer_loss_x(nn.Module):  # 复数 MSE
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        # Ez_loss_conj = (((outputs[0][:,0] - labels[0][:,0]) ** 2 + (outputs[0][:,1] - labels[0][:,1]) ** 2) ** 0.5).mean()
        Ez_loss_conj = ((((outputs[0][:,0] - labels[0][:,1]) ** 2 + (outputs[0][:,1] - labels[0][:,0]) ** 2) ** 0.5) / ((labels[0][:,1] ** 2 + labels[0][:,0] ** 2) ** 0.5)).mean()
        # Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], labels[0][:,0]))
        loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj + loss_rp * self.alpha_chamfer_loss

        return [loss, Ez_loss_conj, loss_rp, -1]


class MSE_Ez_conj(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], -1*labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], -1*labels[0][:,0]))
        # loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj # + loss_rp * self.alpha_chamfer_loss

        return [loss, Ez_loss_conj, -1, -1]

class MSE_Ez_j(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.MSE_loss_fn = nn.functional.mse_loss
        self.chamfer_loss_fn = chamfer_loss_x
        self.orient_cos_loss_fn = calculate_angle_loss
        self.alpha_chamfer_loss = h['alpha_chamfer_loss']

    def __call__(self, outputs, labels):  # [Ez_loss, chamfer_loss, orient_cos_loss]
        # [信号； 环境点云-预测散射点；真实散射点-预测散射点]
        # outputs: [Ez_ReIm, scatters_xyz]
        # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
        # 在__call__方法中执行损失计算
        Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], -1*labels[0][:,0]))
        # loss_rp = self.chamfer_loss_fn(outputs[1], labels[1])
        loss = Ez_loss_conj # + loss_rp * self.alpha_chamfer_loss

        # # NMSE
        NMSE = (((outputs[0][:,0] - labels[0][:,1]) ** 2 + (outputs[0][:,1] + labels[0][:,0]) ** 2) / (labels[0][:,0] ** 2 + labels[0][:,1] ** 2 + 1e-8)).mean()
        loss = NMSE

        return [loss, Ez_loss_conj, -1, -1]


class ComputeLoss:
    def __init__(self, model_hyp):
        h = model_hyp  # hyperparameters
        loss_dict = {
            'MSE_Ez_conj___chamfer_loss___orient_cos_loss': MSE_Ez_conj___chamfer_loss___orient_cos_loss,
            'MSE_Ez_reverse___chamfer_loss___orient_cos_loss': MSE_Ez_reverse___chamfer_loss___orient_cos_loss,
            'MSE_Ez_reverse___chamfer_loss_x___orient_cos_loss': MSE_Ez_reverse___chamfer_loss_x___orient_cos_loss,
            'MSE_Ez_reverse___chamfer_loss_x___orient_cos_loss_Ez_theta': MSE_Ez_reverse___chamfer_loss_x___orient_cos_loss_Ez_theta,
            'MSE_Ez_reverse___chamfer_loss_x': MSE_Ez_reverse___chamfer_loss_x,
            'MSE_Ez___chamfer_loss_x': MSE_Ez___chamfer_loss_x,
            'MSE_Ez_conj___chamfer_loss_x': MSE_Ez_conj___chamfer_loss_x,
            'MSE_Ez_j___chamfer_loss_x': MSE_Ez_j___chamfer_loss_x,
            'MSE_Ez': MSE_Ez,
            'MSE_Ez_fu': MSE_Ez_fu,
            'MSE_Ez_conj': MSE_Ez_conj,
            'MSE_Ez_j': MSE_Ez_j,
            'MSE_Ez_reverse___chamfer_loss_x_ckai': MSE_Ez_reverse___chamfer_loss_x_ckai,
            'MSE_Ez_reverse_ReIm___chamfer_loss_x': MSE_Ez_reverse_ReIm___chamfer_loss_x
            # 'MSE_Ez___chamfer_loss___orient_cos_loss': MSE_Ez_conj___chamfer_loss
        }
        loss_type = h['loss_type']
        if loss_type in loss_dict.keys():
            self.loss_fn = loss_dict[loss_type](h)

    def __call__(self, outputs, labels):
        # 在__call__方法中执行损失计算
        loss = self.loss_fn(outputs, labels)
        return loss
