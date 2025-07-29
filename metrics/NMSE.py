import os
import torch
import numpy as np

def compute_NMSE(labels, outputs):
    eps = 1e-8
    NMSE = (((outputs[:, 0] - labels[:, 0]) ** 2 + (outputs[:, 1] - labels[:, 1]) ** 2).mean() / (
                labels[:, 0] ** 2 + labels[:, 1] ** 2 + eps).mean())
    return NMSE


def compute_NMSE_Ez_j(labels, outputs):
    eps = 1e-8
    #         Ez_loss_conj = 0.5*(self.MSE_loss_fn(outputs[0][:,0], labels[0][:,1]) + self.MSE_loss_fn(outputs[0][:,1], -1*labels[0][:,0]))
    NMSE = (((outputs[:, 0] - labels[:, 1]) ** 2 + (outputs[:, 1] + labels[:, 0]) ** 2).mean() / (
                labels[:, 0] ** 2 + labels[:, 1] ** 2).mean())
    return NMSE


def compute_NMSE_Ez_fu(labels, outputs):
    eps = 1e-8
    #      (outputs[0][:,0], -1 * labels[0][:,0]) + self.MSE_loss_fn(outputs[0][:,1], -1 * labels[0][:,1]))
    NMSE = (((outputs[:, 0] + labels[:, 0]) ** 2 + (outputs[:, 1] + labels[:, 1]) ** 2).mean() / (
                labels[:, 0] ** 2 + labels[:, 1] ** 2).mean())
    return NMSE


