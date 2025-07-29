import torch
from tqdm import tqdm
import time
import argparse
import numpy as np
from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from utils.general import TQDM_BAR_FORMAT, set_logging, Profile, parse_yaml, LoadPCS, parse_pcd_param, pc_denormalize
# LOGGER,
from utils.callbacks import Callbacks
from utils.plots import plot_end_train_scatters, plot_end_train_Ez
# from utils.metrics import ConfusionMatrix

from metrics.NMSE import compute_NMSE, compute_NMSE_Ez_j, compute_NMSE_Ez_fu


# FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径 train.py main.py
# ROOT = FILE.parents[0]  # YOLOv5 root directory ROOT
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def run(batch_size,
        half=False, # amp
        model=None,
        dataloader=None,
        save_dir=None,
        plots=False,
        callbacks=Callbacks(),
        compute_loss=None,
        device=None,
        LOGGER=None,
        training=False,
        data_yml=None,
        pcd_file_name=None,
        epoch=None,
        reverse_conj=None,
        end_train_plot=None,
        plot_scatters=None,
        if_Ez_theta=False,
        if_s_pattern=False,
        scatter_BSDF=None,
        if_sh_module=None,
        deg=0,
        block=False,
        rcube_=None,
        ckai_=None):


    t0 = time.time()
    nb = len(dataloader)  # number of batches
    totalloss = 0.0
    mloss = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
    dt = Profile()
    seen = 0

    pbar = enumerate(dataloader)

    # Logger
    LOGGER.info(('%11s' * 6) % (' ', 'GPU_mem', 'l_total', 'l_Ez', 'l_chamfer', 'l_scatter'))
    pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

    data_yaml = parse_yaml(data_yml)
    data_rootdir = data_yaml['data_path']
    pcd_file_path = os.path.join(data_rootdir, pcd_file_name)
    true_pcs, centroid, m = LoadPCS(batch_size, pcd_file_path)

    if end_train_plot:
        # 用于存储所有批次的预测结果和标签
        Ez_pred = []
        Ez_gt = []

    NMSE = 0.0

    # pbar 已经是可迭代
    for batch_i, (inputs, labels) in pbar:
        callbacks.run('on_val_batch_start')
        # tloc, rloc
        inputs = [input.to(device) for input in inputs]  # [tloc, rloc]

        # Inference
        with dt:
            with torch.no_grad():
                if scatter_BSDF:
                    pcd_param_path = os.path.join(data_rootdir, scatter_BSDF)
                    pcd_param = parse_pcd_param(pcd_param_path)  # [n_points, 4]  [pcd_areas 1, pcd_normals 3]
                    # pcd_param = torch.tensor(pcd_param)
                    pcd_param = torch.tile(torch.tensor(pcd_param), (inputs[0].shape[0], 1, 1))
                    pcd_xyz = torch.tensor(pc_denormalize(true_pcs, centroid, m))[0:inputs[0].shape[0]+1, ...]
                    pred = model(inputs[0], inputs[1], if_Ez_theta, if_s_pattern, pcd=[pcd_xyz, pcd_param])
                elif block:
                    block_pcd = np.loadtxt(pcd_file_path)
                    block_pcd = np.tile(np.expand_dims(block_pcd, axis=0), (min(batch_size, inputs[0].shape[0]), 1, 1))
                    if rcube_ is not None:
                        pred = model(inputs[0], inputs[1], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg,
                                     block_pcd=block_pcd, rcube_=rcube_, ckai_=ckai_)
                    else:
                        pred = model(inputs[0], inputs[1], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg,
                                     block_pcd=block_pcd)
                else:
                    pred = model(inputs[0], inputs[1], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg)  # forward

        if end_train_plot:
            ### Dataset_inputs: [tloc, rloc]
            ### Dataset_labels [gtot, scat_gt]
            ### input : [tloc, rloc]
            ### output: [gtot, rcube_, rcube, ckai]
            Ez_pred.append(pred[0])
            Ez_gt.append(labels[0])
            if plot_scatters:
                scatter_pred = pred[2].cpu().detach().numpy()
                scatter_gt = labels[1].cpu().detach().numpy()
                ckai = pred[3].cpu().detach().numpy()
                plot_end_train_scatters(tloc=inputs[0].cpu().detach().numpy(), rloc=inputs[1].cpu().detach().numpy(),
                                        scatter_pred=scatter_pred, scatter_gt=scatter_gt, ckai=ckai, test_index=batch_i*batch_size, save_dir=save_dir)

        if true_pcs.shape[0] != len(inputs[0]):
            true_pcs_label = true_pcs[0:len(inputs[0]), ...]
        else:
            true_pcs_label = true_pcs

        # Loss
        if compute_loss:
            labels = [label.to(device) for label in labels]
            loss = compute_loss(pred[0:2], [labels[0], torch.tensor(true_pcs_label, dtype=torch.float32).to(device),
                                            [labels[1], inputs[1]]])  # loss scaled by batch_size ？？
            # loss: [loss, loss_rp, orient_cos_loss]
            loss_sum = loss[0].sum()

        # if not training:  #  if end_train_plot:
        NMSE = (NMSE * batch_i + compute_NMSE(labels[0], pred[0])) / (batch_i + 1)

        # # # NMSE   MSE_Ez_j
        # NMSE = (NMSE * batch_i + compute_NMSE_Ez_j(labels[0], pred[0])) / (batch_i + 1)

        # # # NMSE   MSE_Ez_fu
        # NMSE = (NMSE * batch_i + compute_NMSE_Ez_fu(labels[0], pred[0])) / (batch_i + 1)


        if not training:
            with open(os.path.join(save_dir, "result_val.txt"), 'a') as f:
                for (rssi, gt_rssi) in zip(labels[0], pred[0]):   # 写反了
                    f.write("{:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(rssi[0],
                                                                          gt_rssi[0],
                                                                          rssi[1],
                                                                          gt_rssi[1]
                                                                          ))

                    # (((outputs[:, 0] - labels[:, 1]) ** 2 + (outputs[:, 1] + labels[:, 0]) ** 2).mean() / (
                    #         labels[:, 0] ** 2 + labels[:, 1] ** 2).mean())
                    # # NMSE   MSE_Ez_j
                    # f.write("{:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(-1 * gt_rssi[1],
                    #                                                       rssi[0],
                    #                                                       gt_rssi[0],
                    #                                                       rssi[1]
                    #                                                       ))

                    # # NMSE   MSE_Ez_fu
                    # f.write("{:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(-1 * gt_rssi[0],
                    #                                                       rssi[0],
                    #                                                   -1 * gt_rssi[1],
                    #                                                       rssi[1]
                    #                                                       ))





        mloss = [(mloss_item * batch_i + loss_item) / (batch_i + 1) for mloss_item, loss_item in zip(mloss, loss)]
        totalloss += loss[0].sum()

        # Metrics
        seen += pred[0].shape[0]

        # mem
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # GPU memory (GB) (GB)
        pbar.set_description(('%11s'+ '%11s' + '%11.4g' * 4) %
                             (' ', mem, mloss[0], mloss[1], mloss[2], mloss[3]))
        callbacks.run('on_val_batch_end')

    LOGGER.info(f"\nNMSE: {NMSE}\n NMSE_dB: {-1 * 10 * torch.log10(NMSE)}")

    if end_train_plot:
        Ez_pred = torch.cat(Ez_pred, dim=0).cpu().detach().numpy()
        Ez_gt = torch.cat(Ez_gt, dim=0).cpu().detach().numpy()
        plot_end_train_Ez(Ez_pred, Ez_gt, reverse_conj, save_dir)


    # Plot images 画前三组数据
    if plots:
        pbar = enumerate(dataloader)
        for i, (inputs_plot, labels_plot) in pbar:
            if i != 0:
                break
            inputs_plot = [input.to(device) for input in inputs_plot]  # [tloc, rloc]
            # pred_plot = model(inputs_plot[0], inputs_plot[1])  # forward
            if block:
                block_pcd = np.loadtxt(pcd_file_path)
                block_pcd = np.tile(np.expand_dims(block_pcd, axis=0), (min(batch_size, inputs_plot[0].shape[0]), 1, 1))

            pred_plot = model(inputs_plot[0], inputs_plot[1], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg,
                              block_pcd=block_pcd, rcube_=rcube_, ckai_=ckai_)

            ### model.forward(self, tloc, rloc))
            ### # inputs: [tloc, rloc]
            ### # labels [gtot, scat_gt]
            ### input : [tloc, rloc]
            ### output: [gtot, rcube_, rcube, ckai]

            callbacks.run('end_epoch_plot', Ez_pred=pred_plot[0].cpu().detach().numpy(), Ez_gt=labels_plot[0].cpu().detach().numpy(),
                          tloc=inputs_plot[0].cpu().detach().numpy(), rloc=inputs_plot[1].cpu().detach().numpy(),
                          scatter_pred=pred_plot[2].cpu().detach().numpy(),
                          scatter_gt=labels_plot[1].cpu().detach().numpy(),
                          ckai=pred_plot[3].cpu().detach().numpy(), epoch=epoch, save_dir=save_dir, reverse_conj=reverse_conj, train_or_val="val")

            # # pazhou 4TX
            # TX_list = [[369, -17, 63], [473, 16, 63], [248, 55, 63], [365, 138, 63]]
            # # TX_list = [[369, -17, 63]]
            # RX_list = [30]
            # n = 99
            # for t_x in TX_list:
            #     for r_x in RX_list:
            #         tx = torch.tensor([t_x] * 1000, device='cuda')
            #         rx = torch.tensor([[0 + 12 * i, 0 + 25 * j, r_x]
            #                            for i in range(50) for j in range(20)], device='cuda')

            # indoor 3cm
            TX_list = [[0, 0, 2], [0, 0, 3], [0, -2.5, 1], [0, -2.5, 2.5], [2.5, -2.5, 2.5], [-2.5, -2.5, 2]]
            # TX_list = [[-2.5, -2.5, 2]]
            RX_list = [0.3 + (4 - 0.3) / 6 * x for x in range(7)]
            n = 49
            for t_x in TX_list:
                for r_x in RX_list:
                    tx = torch.tensor([t_x] * 2500, device='cuda')
                    rx = torch.tensor([[-4.8 + 10 / n * i, -4.7 + 9.2 / n * j, r_x]
                                       for i in range(50) for j in range(50)], device='cuda')
                    # plot_Ez_fine_sample_scatter_contourf(tx, rx, n)


            # # cube fine sample
            # TX_list = [[0.3, 0.3, 0], [0.3, 0.3, 0.2], [0.2250, 0, 0], [0.2250, 0, 0.2], [0.45, 0, 0], [0.45, 0, 0.2]]
            # RX_list = [-0.3, -0.15, -0.0750, 0, 0.0750, 0.15, 0.3]
            # # RX_list = [-0.3, 0, 0.3]
            # n = 30
            # for t_x in TX_list:
            #     for r_x in RX_list:
            #         tx = torch.tensor([t_x] * n ** 2, device='cuda')
            #         rx = torch.tensor([[-0.5 + 1 / n * i, -0.5 + 1 / n * j, r_x]
            #                            for i in range(30) for j in range(30)], device='cuda')
            #         # plot_Ez_fine_sample_scatter_contourf(tx, rx, n)

            # # sinna_PKU
            # # TX_list = [[0, 0, 40], [0, -300, 40]]
            # TX_list = [[0, 0, 40]]
            # RX_list = [1]
            # n = 76800
            # n_x = 240 - 1
            # n_y = 320 - 1
            # for t_x in TX_list:
            #     for r_x in RX_list:
            #         tx = torch.tensor([t_x] * (n_x + 1) * (n_y + 1), device='cuda')
            #         rx = torch.tensor([[-600 + 1200 / n_x * i, -800 + 1600 / n_y * j, r_x]
            #                   for i in range(n_x + 1) for j in range(n_y + 1)], device='cuda')



                    # t_x = [0.3, 0.3, 0]
                    # n = 30
                    # # tx = torch.tensor([[0.45, 0, 0]] * 16, device='cuda')
                    # # tx = torch.tensor([[0.2250, 0, 0]] * 900, device='cuda')
                    # tx = torch.tensor([t_x] * n ** 2, device='cuda')
                    # rx = torch.tensor([[-0.5 + 1 / n * i, -0.5 + 1 / n * j, 0.0750]
                    #                    for i in range(30) for j in range(30)], device='cuda')

                    # pred = model(tx, rx, if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg)  # forward

                    # # 计算颜色值，排除异常
                    # color = torch.sqrt(pred[0][:, 0] ** 2 + pred[0][:, 1] ** 2)
                    # color = torch.nan_to_num(color)
                    # color = color.cpu().numpy()

                    # block 1000
                    block_pcd = np.loadtxt(pcd_file_path)
                    color = []
                    color_Re = []
                    color_Im = []
                    bz = 100 # 300  # 96
                    for i in range(int(tx.shape[0] / bz)):
                        block_pcd_val = np.tile(np.expand_dims(block_pcd, axis=0), (bz, 1, 1))
                        # pred = model(tx[i*bz: (i+1)*bz, :], rx[i*bz: (i+1)*bz, :], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd_val) # block

                        with torch.no_grad():
                            pred = model(tx[i * bz: (i + 1) * bz, :], rx[i * bz: (i + 1) * bz, :], if_Ez_theta,
                                         if_s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd_val,
                                         rcube_=rcube_, ckai_=ckai_)  # block

                        pred = [pred.cpu().detach() for pred in pred]
                        color_pred = torch.sqrt(pred[0][:, 0] ** 2 + pred[0][:, 1] ** 2)
                        color_pred_Re = pred[0][:, 0]
                        color_pred_Im = pred[0][:, 1]
                        color_pred = torch.nan_to_num(color_pred)
                        # color_pred = color_pred.cpu().detach().numpy()
                        color.append(color_pred)
                        color_Re.append(color_pred_Re)
                        color_Im.append(color_pred_Im)
                    color = np.array(color).reshape(-1)
                    color_Re = np.array(color_Re).reshape(-1)
                    color_Im = np.array(color_Im).reshape(-1)



                    # # plt.scatter(rx[:, 0].cpu(), rx[:, 1].cpu(), c=color, s=1) # sionna s=1
                    # plt.scatter(rx[:, 0].cpu(), rx[:, 1].cpu(), c=color, s=30)  # indoor bar 3cm s=30
                    # plt.colorbar(label="Color Intensity")  # 添加颜色条
                    # plt.savefig(os.path.join(save_dir, f"TX_{t_x}_points_{n}_scatter_RX_xyplane_z_{rx[-1, -1]:.3f}.png"), dpi=300)
                    # # plt.show()
                    # plt.close()

                    x = np.array(rx[:, 0].cpu())
                    y = np.array(rx[:, 1].cpu())
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()

                    amplitude = np.array(color)

                    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

                    grid_amplitude = griddata((x, y), amplitude, (grid_x, grid_y), method='cubic')  #

                    plt.figure(figsize=(8, 6))
                    contour = plt.contourf(grid_x, grid_y, grid_amplitude, levels=100, cmap='viridis')
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.colorbar(contour, label='Amplitude')
                    plt.title('Amplitude Contour')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.savefig(
                        os.path.join(save_dir, f"TX_{t_x}_points_{n}_contour_RX_xyplane_z_{rx[-1, -1]:.3f}_epoch{epoch}_abs.png"),
                        dpi=300)
                    # plt.show()
                    plt.close()


                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, y, c=amplitude, s=0.5) # PKU_sionna s=1
                    # plt.scatter(x, y, c=amplitude, s=30) # indoor bar s=30
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.colorbar()
                    plt.title('Amplitude sctter')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.savefig(
                        os.path.join(save_dir, f"TX_{t_x}_points_{n}_scatter_RX_xyplane_z_{rx[-1, -1]:.3f}_epoch{epoch}_abs.png"),
                        dpi=300)
                    # plt.show()
                    plt.close()





                    amplitude = np.array(color_Re)

                    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

                    grid_amplitude = griddata((x, y), amplitude, (grid_x, grid_y), method='cubic')  #

                    plt.figure(figsize=(8, 6))
                    contour = plt.contourf(grid_x, grid_y, grid_amplitude, levels=100, cmap='viridis')
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.colorbar(contour, label='Amplitude')
                    plt.title('Amplitude Contour')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.savefig(
                        os.path.join(save_dir, f"TX_{t_x}_points_{n}_contour_RX_xyplane_z_{rx[-1, -1]:.3f}_epoch{epoch}_Re.png"),
                        dpi=300)
                    # plt.show()
                    plt.close()


                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, y, c=amplitude, s=0.5) # PKU_sionna s=1
                    # plt.scatter(x, y, c=amplitude, s=30) # indoor bar s=30
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.colorbar()
                    plt.title('Amplitude sctter')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.savefig(
                        os.path.join(save_dir, f"TX_{t_x}_points_{n}_scatter_RX_xyplane_z_{rx[-1, -1]:.3f}_epoch{epoch}_Re.png"),
                        dpi=300)
                    # plt.show()
                    plt.close()




                    amplitude = np.array(color_Im)

                    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

                    grid_amplitude = griddata((x, y), amplitude, (grid_x, grid_y), method='cubic')  #

                    plt.figure(figsize=(8, 6))
                    contour = plt.contourf(grid_x, grid_y, grid_amplitude, levels=100, cmap='viridis')
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.colorbar(contour, label='Amplitude')
                    plt.title('Amplitude Contour')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.savefig(
                        os.path.join(save_dir, f"TX_{t_x}_points_{n}_contour_RX_xyplane_z_{rx[-1, -1]:.3f}_epoch{epoch}_Im.png"),
                        dpi=300)
                    # plt.show()
                    plt.close()


                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, y, c=amplitude, s=0.5) # PKU_sionna s=1
                    # plt.scatter(x, y, c=amplitude, s=30) # indoor bar s=30
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.colorbar()
                    plt.title('Amplitude sctter')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.savefig(
                        os.path.join(save_dir, f"TX_{t_x}_points_{n}_scatter_RX_xyplane_z_{rx[-1, -1]:.3f}_epoch{epoch}_Im.png"),
                        dpi=300)
                    # plt.show()
                    plt.close()


                    #----------------------------------------------------------------------------------

                    # grid_amplitude = griddata((x, y), color_Re, (grid_x, grid_y), method='cubic')  #
                    #
                    # plt.figure(figsize=(8, 6))
                    # contour = plt.contourf(grid_x, grid_y, grid_amplitude, levels=100, cmap='viridis')
                    # plt.colorbar(contour, label='Amplitude')
                    # plt.title('Amplitude Contour')
                    # plt.xlabel('X')
                    # plt.ylabel('Y')
                    # plt.savefig(
                    #     os.path.join(save_dir, f"TX_{t_x}_points_{n}_contour_RX_xyplane_z_{rx[-1, -1]:.3f}_Re_epoch{epoch}.png"),
                    #     dpi=300)
                    # # plt.show()
                    # plt.close()
                    #
                    #
                    # grid_amplitude = griddata((x, y), color_Im, (grid_x, grid_y), method='cubic')  #
                    #
                    # plt.figure(figsize=(8, 6))
                    # contour = plt.contourf(grid_x, grid_y, grid_amplitude, levels=100, cmap='viridis')
                    # plt.colorbar(contour, label='Amplitude')
                    # plt.title('Amplitude Contour')
                    # plt.xlabel('X')
                    # plt.ylabel('Y')
                    # plt.savefig(
                    #     os.path.join(save_dir, f"TX_{t_x}_points_{n}_contour_RX_xyplane_z_{rx[-1, -1]:.3f}_Im_epoch{epoch}.png"),
                    #     dpi=300)
                    # # plt.show()
                    # plt.close()



    # Print results
    LOGGER.debug(('%11s' + '%11s' + '%11.4g' * 4) % (' ', mem, mloss[0], mloss[1], mloss[2], mloss[3]))

    final_result = [f'mloss_total: {mloss[0]}, mloss_Ez {mloss[1]}, mloss_chamfer {mloss[2]}, mloss_scatter {mloss[3]}']
    LOGGER.info(f"Mean MSE loss: {str(final_result)} ")

    # Print speeds
    t = dt.t / seen * 1E3  # speeds per TxRx  /ms
    if not training:
        LOGGER.info(f'Speed: %.1fms inference per microdata' % t)

    callbacks.run('on_val_end')

    LOGGER.debug(f'Val end! Time: {time.time() - t0:.3f}s, Loss: {str(final_result)}')

    return mloss



