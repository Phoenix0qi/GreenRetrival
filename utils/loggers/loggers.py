import os.path

from utils.general import colorstr
from utils.plots import plot_kai_scatter, plot_signal, plot_results

import numpy as np

class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        # self.plot = not opt.noplots  # plot results
        self.csv = True

    def on_train_start(self):
        self.logger.debug(f"On train start!")

    def on_pretrain_routine_start(self):
        self.logger.debug(f"On pretrain routine start!")

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        self.logger.info(f"Pretrain routine ends!")

    def on_train_epoch_start(self):
        self.logger.debug(f"On train epoch start!")

    def on_train_batch_start(self):
        self.logger.debug(f"On train batch start!")

    def on_train_batch_end(self, mloss): # mloss [18,2]
        self.logger.debug(f"On train batch end! f'mloss_total: {mloss[0]}, mloss_Ez {mloss[1]}, mloss_chamfer {mloss[2]}, mloss_scatter {mloss[3]}")        # self.logger.debug(f"On train batch end! Mean loss: {mloss}, mean Accuracy: {acc}.")

    def end_epoch_plot(self, Ez_pred, Ez_gt, tloc, rloc, scatter_pred, scatter_gt, ckai, epoch, save_dir, reverse_conj, train_or_val):
        # Callback runs on train epoch end
        # 根据epoch 画图 and 存储 # 画图：散射点-信号；
        # input: Ez_pred, Ez_gt: [batch, 2] Re-Im
        # tloc, rloc, scatter_pred: [batch, 3], [batch, n_scatters, 3]
        # scatter_gt, ckai: [batch, 10, 9], [batch, n_scatters, 2]
        self.logger.debug(f"On train epoch end! Current epoch: {epoch}.")
        save_dir = os.path.join(save_dir,  "snapshot")
        os.makedirs(save_dir, exist_ok=True)

        if reverse_conj == 0:
            Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = Ez_pred[:, 0], Ez_pred[:, 1], np.sqrt(Ez_pred[:, 0]**2 + Ez_pred[:, 1]**2)
            Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0]**2 + Ez_gt[:, 1]**2)
        elif reverse_conj == 1: # conj
            Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = -1*Ez_pred[:, 1], -1*Ez_pred[:, 0], np.sqrt(Ez_pred[:, 0]**2 + Ez_pred[:, 1]**2)
            Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0]**2 + Ez_gt[:, 1]**2)
        elif reverse_conj == 3:  # label*j
            Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = -1*Ez_pred[:, 1], Ez_pred[:, 0], np.sqrt(Ez_pred[:, 0]**2 + Ez_pred[:, 1]**2)
            Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0]**2 + Ez_gt[:, 1]**2)
        else:     # 2 reverse
            Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = Ez_pred[:, 1], Ez_pred[:, 0], np.sqrt(Ez_pred[:, 0]**2 + Ez_pred[:, 1]**2)
            Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0]**2 + Ez_gt[:, 1]**2)

        # plot_signal(pred, gt, title, savedir, filename)
        plot_signal(Ez_pred_Re, Ez_gt_Re, "Re_Ez", save_dir, f"Re_Ez_{train_or_val}_epoch_" + str(epoch))
        plot_signal(Ez_pred_Im, Ez_gt_Im, "Im_Ez", save_dir, f"Im_Ez_{train_or_val}_epoch_" + str(epoch))
        plot_signal(Ez_pred_abs, Ez_gt_abs, "abs_Ez", save_dir, f"abs_Ez_{train_or_val}_epoch_" + str(epoch))

        if abs(ckai).mean() > 1e-8:
            plot_kai_scatter(tloc[0, ...], rloc[0, ...], ckai[0, :, 0], scatter_pred[0, ...], scatter_gt[0, ...],
                             save_dir, f"scatter_Re_{train_or_val}_epoch_" + str(epoch))
            plot_kai_scatter(tloc[0, ...], rloc[0, ...], ckai[0, :, 1], scatter_pred[0, ...], scatter_gt[0, ...],
                             save_dir, f"scatter_Im_{train_or_val}_epoch_" + str(epoch))


    def on_model_save(self, last, epoch, final_epoch):
        self.logger.debug(f"On model save! Last epoch: {last}, current epoch: {epoch}, final epoch: {final_epoch}.")

    def on_train_end(self, best_epoch, best, epoch, loss_best, noplots):
        # Callback runs on training end, i.e. saving best models
        self.logger.info(f"On train end! Best epoch: {best_epoch}/{epoch}, final best mloss on val: {loss_best}, best model is saved to {best}.")

        if not noplots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png

        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def on_val_batch_start(self):
        self.logger.debug(f"On val batch start!")

    def on_val_batch_end(self):
        self.logger.debug(f"On val batch end!")

    def on_val_end(self):
        self.logger.debug(f"On val end!")

    def on_fit_epoch_end(self, log_vals, log_trains, epoch):
        # log_vals = [val_loss, lr]
        # log_trains = mloss
        self.logger.debug(f"On fit epoch end!")
        x = {"train_mloss_total": float(log_trains[0]), "train_mloss_Ez": float(log_trains[1]), "train_mloss_chamfer": float(log_trains[2]), "train_mloss_scatter": float(log_trains[3]),
             "val_loss_total": float(log_vals[0][0]), "val_mloss_Ez": float(log_vals[0][1]), "val_loss_chamfer": float(log_vals[0][2]), "val_loss_scatter": float(log_vals[0][3]),
             # "lr-0": log_vals[1][0], "lr-1": log_vals[1][1], "lr-2": log_vals[1][2]}
             "lr-0": log_vals[1][0]}
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + list(x.keys()))).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.7g,' * n % tuple([epoch] + list(x.values()))).rstrip(',') + '\n')