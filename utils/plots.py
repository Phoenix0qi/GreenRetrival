import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


def plot_signal(pred, gt, title, savedir, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(gt, label='Ground Truth')
    plt.plot(pred, label='Prediction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(savedir, filename), dpi=300)
    plt.close()

def plot_kai_scatter(tloc, rloc, kai, r, scatter_gt, savedir, filename):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(tloc[0], tloc[1], tloc[2], s=100, marker='.', c='r', label='P_Tx')
    p = ax.scatter(rloc[0], rloc[1], rloc[2], s=100, marker='.', c='b', label='P_Rx')
    p = ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=10, marker='.', c=kai, cmap='rainbow')

    if scatter_gt.mean() + 1 > 1e-3:  # indoor 画 pazhou 不画
        for i in range(9):
            p = ax.scatter(scatter_gt[i, 3], scatter_gt[i, 4], scatter_gt[i, 5], s=100, marker='.', c='g', label='scatter_gt')

        ax.set_xlim(-9, -5)
        ax.set_ylim(2, 11)
        ax.set_zlim(0, 2.7)
    # else:
    #     ax.set_xlim(-9, -5)
    #     ax.set_ylim(2, 11)
    #     ax.set_zlim(0, 2.7)

    fig.colorbar(p, ax=ax)
    ax.legend()

    plt.savefig(os.path.join(savedir, filename), dpi=300)
    plt.close()

def plot_end_train_scatters(tloc, rloc, scatter_pred, scatter_gt, ckai, test_index, save_dir):
    n_frame = scatter_pred.shape[0]
    save_dir = os.path.join(save_dir, "final_scatter")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_frame):
        plot_kai_scatter(tloc[i, ...], rloc[i, ...], ckai[i, :, 0], scatter_pred[i, ...], scatter_gt[i, ...],
                         save_dir, "scatter_Re_frame_" + str(test_index + i))
        plot_kai_scatter(tloc[i, ...], rloc[i, ...], ckai[i, :, 1], scatter_pred[i, ...], scatter_gt[i, ...],
                         save_dir, "scatter_Im_frame_" + str(test_index + i))


def plot_end_train_Ez(Ez_pred, Ez_gt, reverse_conj, save_dir):
    save_dir = os.path.join(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if reverse_conj == 0:
        Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = Ez_pred[:, 0], Ez_pred[:, 1], np.sqrt(
            Ez_pred[:, 0] ** 2 + Ez_pred[:, 1] ** 2)
        Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0] ** 2 + Ez_gt[:, 1] ** 2)
    elif reverse_conj == 1:
        Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = -1 * Ez_pred[:, 1], -1 * Ez_pred[:, 0], np.sqrt(
            Ez_pred[:, 0] ** 2 + Ez_pred[:, 1] ** 2)
        Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0] ** 2 + Ez_gt[:, 1] ** 2)
    elif reverse_conj == 3:  # label*j
        Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = -1 * Ez_pred[:, 1], Ez_pred[:, 0], np.sqrt(
            Ez_pred[:, 0] ** 2 + Ez_pred[:, 1] ** 2)
        Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0] ** 2 + Ez_gt[:, 1] ** 2)
    else:
        Ez_pred_Re, Ez_pred_Im, Ez_pred_abs = Ez_pred[:, 1], Ez_pred[:, 0], np.sqrt(
            Ez_pred[:, 0] ** 2 + Ez_pred[:, 1] ** 2)
        Ez_gt_Re, Ez_gt_Im, Ez_gt_abs = Ez_gt[:, 0], Ez_gt[:, 1], np.sqrt(Ez_gt[:, 0] ** 2 + Ez_gt[:, 1] ** 2)

    # plot_signal(pred, gt, title, savedir, filename)
    plot_signal(Ez_pred_Re, Ez_gt_Re, "Re_Ez", save_dir, "Re_Ez_val")
    plot_signal(Ez_pred_Im, Ez_gt_Im, "Im_Ez", save_dir, "Im_Ez_val")
    plot_signal(Ez_pred_abs, Ez_gt_abs, "abs_Ez", save_dir, "abs_Ez_val")
    plot_signal(Ez_pred_Re[0:100, ...], Ez_gt_Re[0:100, ...], "Re_Ez", save_dir, "Re_Ez_100_val")
    plot_signal(Ez_pred_Im[0:100, ...], Ez_gt_Im[0:100, ...], "Im_Ez", save_dir, "Im_Ez_100_val")
    plot_signal(Ez_pred_abs[0:100, ...], Ez_gt_abs[0:100, ...], "abs_Ez", save_dir, "abs_Ez_100_val")

def plot_results(file='path/to/results.csv', dir=''):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    # csv = [epoch, train_loss, train_acc, val_loss, val_acc, lr]
    # csv = [epoch, train_loss, train_loss_x, train_loss_y, val_loss, val_loss_x, val_loss_y, test_loss, lr[0], lr[1], lr[2]]
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(15, 15), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 9, 5, 6, 7, 8, 10]): # 调整画图和csv的列的顺序
                y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ':', label='smooth', linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)

        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=300)
    plt.close()


def plot_BSDF_s_pattern_2D():
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    # 参数
    k = 1.0  # 波数
    pcd_nearest_param = torch.tensor([1.0], dtype=torch.float32)  # 假设为标量
    theta_in = torch.tensor(45.0, dtype=torch.float32) * torch.pi / 180  # 入射角 θ_in (弧度制)
    phi_in = torch.tensor(90.0, dtype=torch.float32) * torch.pi / 180  # 入射角 φ_in (弧度制)

    # 入射向量 (in_local)
    in_local = torch.tensor([
        torch.sin(theta_in) * torch.cos(phi_in),
        torch.sin(theta_in) * torch.sin(phi_in),
        torch.cos(theta_in)
    ], dtype=torch.float32)

    # 散射角网格
    theta_sc = torch.linspace(0, torch.pi, 180)  # θ_sc in [0, π]
    phi_sc = torch.linspace(0, 2 * torch.pi, 360)  # φ_sc in [0, 2π]
    theta_sc_grid, phi_sc_grid = torch.meshgrid(theta_sc, phi_sc, indexing="ij")

    # 散射向量 (sc_local) in Cartesian coordinates
    sc_local = torch.stack([
        torch.sin(theta_sc_grid) * torch.cos(phi_sc_grid),
        torch.sin(theta_sc_grid) * torch.sin(phi_sc_grid),
        torch.cos(theta_sc_grid)
    ], dim=-1)  # Shape: (180, 360, 3)

    # Safe sinc function
    def safe_sinc(x):
        return torch.where(torch.abs(x) < 1e-6, torch.ones_like(x), torch.sin(x) / x)

    # 计算 Spq
    sinc_x = safe_sinc((k * pcd_nearest_param[..., 0] ** 0.5 / 2 * (sc_local[..., 0] - in_local[0])) + 1e-8)
    sinc_y = safe_sinc((k * pcd_nearest_param[..., 0] ** 0.5 / 2 * (sc_local[..., 1] - in_local[1])) + 1e-8)
    in_local_dot_sc_local = torch.sum(in_local * sc_local, dim=-1)  # 点积

    Spq = torch.abs(sinc_x * sinc_y * pcd_nearest_param[..., 0] / torch.pi ** 0.5 * in_local[
        2] * in_local_dot_sc_local)  # Shape: (180, 360)

    # 转为 NumPy 格式以便可视化
    Spq_numpy = Spq.cpu().numpy()

    # 绘制图像
    plt.figure(figsize=(10, 5))
    plt.imshow(Spq_numpy, extent=[0, 360, 0, 180], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Spq Intensity")
    plt.xlabel("φ_sc (degrees)")
    plt.ylabel("θ_sc (degrees)")
    plt.title("Spq Distribution (Fixed Incident Angle)")
    plt.show()

def plot_BSDF_s_pattern_3D():
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 参数
    k = 1.0  # 波数
    pcd_nearest_param = torch.tensor([1.0], dtype=torch.float32)  # 假设为标量
    theta_in = torch.tensor(45.0, dtype=torch.float32) * torch.pi / 180  # 入射角 θ_in (弧度制)
    phi_in = torch.tensor(90.0, dtype=torch.float32) * torch.pi / 180  # 入射角 φ_in (弧度制)

    # 入射向量 (in_local)
    in_local = torch.tensor([
        torch.sin(theta_in) * torch.cos(phi_in),
        torch.sin(theta_in) * torch.sin(phi_in),
        torch.cos(theta_in)
    ], dtype=torch.float32)

    # 散射角网格
    theta_sc = torch.linspace(0, torch.pi, 180)  # θ_sc in [0, π]
    phi_sc = torch.linspace(0, 2 * torch.pi, 360)  # φ_sc in [0, 2π]
    theta_sc_grid, phi_sc_grid = torch.meshgrid(theta_sc, phi_sc, indexing="ij")

    # 散射向量 (sc_local) in Cartesian coordinates
    sc_local = torch.stack([
        torch.sin(theta_sc_grid) * torch.cos(phi_sc_grid),
        torch.sin(theta_sc_grid) * torch.sin(phi_sc_grid),
        torch.cos(theta_sc_grid)
    ], dim=-1)  # Shape: (180, 360, 3)

    # Safe sinc function
    def safe_sinc(x):
        return torch.where(torch.abs(x) < 1e-6, torch.ones_like(x), torch.sin(x) / x)

    # 计算 Spq
    sinc_x = safe_sinc((k * pcd_nearest_param[..., 0] ** 0.5 / 2 * (sc_local[..., 0] - in_local[0])) + 1e-8)
    sinc_y = safe_sinc((k * pcd_nearest_param[..., 0] ** 0.5 / 2 * (sc_local[..., 1] - in_local[1])) + 1e-8)
    in_local_dot_sc_local = torch.sum(in_local * sc_local, dim=-1)  # 点积

    Spq = torch.abs(sinc_x * sinc_y * pcd_nearest_param[..., 0] / torch.pi ** 0.5 * in_local[
        2] * in_local_dot_sc_local)  # Shape: (180, 360)

    # 球坐标 -> 笛卡尔坐标
    r = Spq  # 把 Spq 的值作为球面半径
    x = r * torch.sin(theta_sc_grid) * torch.cos(phi_sc_grid)
    y = r * torch.sin(theta_sc_grid) * torch.sin(phi_sc_grid)
    z = r * torch.cos(theta_sc_grid)

    # 转为 NumPy 格式以便可视化
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    z_np = z.cpu().numpy()

    # 3D 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_np, y_np, z_np, facecolors=plt.cm.viridis(Spq.cpu().numpy() / Spq.max()), rstride=2, cstride=2,
                    alpha=0.9)

    # 设置图形属性
    ax.set_title("3D Directional Distribution (Fixed Incident Angle)", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_Ez_fine_contourf():
    from mpl_toolkits.mplot3d import Axes3D

    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import numpy as np

    t_x = [0.45, 0, 0]
    n = 30
    # tx = torch.tensor([[0.45, 0, 0]] * 16, device='cuda')
    # tx = torch.tensor([[0.2250, 0, 0]] * 900, device='cuda')
    tx = torch.tensor([t_x] * n**2, device='cuda')
    rx = torch.tensor([[-0.5 + 1 / n * i, -0.5 + 1 / n * j, 0.1125]
                       for i in range(30) for j in range(30)], device='cuda')
    pred = model(tx, rx, if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg)  # forward



    # 计算颜色值，排除异常
    color = torch.sqrt(pred[0][:, 0] ** 2 + pred[0][:, 1] ** 2)
    color = torch.nan_to_num(color)  # 替换 NaN 为 0
    color = color.cpu().numpy()  # 转换为 numpy 数组以供 Matplotlib 使用
    # 绘制散点图
    plt.scatter(rx[:, 0].cpu(), rx[:, 1].cpu(), c=color)
    plt.colorbar(label="Color Intensity")  # 添加颜色条
    plt.show()

    x = np.array(rx[:, 0].cpu())
    y = np.array(rx[:, 1].cpu())
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    amplitude = np.array(color)

    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]  # 生成 100x100 网格

    # 插值数据到网格
    grid_amplitude = griddata((x, y), amplitude, (grid_x, grid_y), method='cubic')  # 插值

    # 绘制等高填充图
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, grid_amplitude, levels=100, cmap='viridis')  #
    plt.colorbar(contour, label='Amplitude')  #
    plt.title('Amplitude Contour')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(save_dir, f"TX_{t_x}_points_{n}"), dpi=300)
    plt.show()












def plot_Ez_fine_sample_scatter_contourf(tx, rx, n):


    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import numpy as np

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
    bz = 20
    for i in range(int(tx.shape[0]/bz)):
        block_pcd_val = np.tile(np.expand_dims(block_pcd, axis=0), (bz, 1, 1))
        # pred = model(tx[i*bz: (i+1)*bz, :], rx[i*bz: (i+1)*bz, :], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd_val) # block
        pred = model(tx[i*bz: (i+1)*bz, :], rx[i*bz: (i+1)*bz, :], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd_val, rcube_=rcube_, ckai_=ckai_) # block

        color_pred = torch.sqrt(pred[0][:, 0] ** 2 + pred[0][:, 1] ** 2)
        color_pred = torch.nan_to_num(color_pred)
        color_pred = color_pred.cpu().numpy()
        color.append(color_pred)
    color = np.array(color).reshape(-1)


    # plt.scatter(rx[:, 0].cpu(), rx[:, 1].cpu(), c=color)
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
    plt.colorbar(contour, label='Amplitude')
    plt.title('Amplitude Contour')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(save_dir, f"TX_{t_x}_points_{n}_contour_RX_xyplane_z_{rx[-1, -1]:.3f}.png"), dpi=300)
    # plt.show()
    plt.close()




    import torch

    TX_list = [[0.3, 0.3, 0], [0.3, 0.3, 0.2], [0.2250, 0, 0], [0.2250, 0, 0.2], [0.45, 0, 0], [0.45, 0, 0.2]]
    RX_list = [-0.15, -0.0750, 0, 0.0750, 0.15]
    n = 30
    for t_x in TX_list:
        for r_x in RX_list:
            tx = torch.tensor([t_x] * n ** 2, device='cuda')
            rx = torch.tensor([[-0.5 + 1 / n * i, -0.5 + 1 / n * j, r_x]
                               for i in range(30) for j in range(30)], device='cuda')
            plot_Ez_fine_sample_scatter_contourf(tx, rx, n)


    TX_list = [[0.3, 0.3, 0], [0.3, 0.3, 0.2], [0.2250, 0, 0], [0.2250, 0, 0.2], [0.45, 0, 0], [0.45, 0, 0.2]]
    RX_list = [-0.15, -0.0750, 0, 0.0750, 0.15]
    n = 29
    for t_x in TX_list:
        for r_x in RX_list:
            tx = torch.tensor([t_x] * 30 ** 2, device='cuda')
            rx = torch.tensor([[-0.5 + 1 / n * i, -0.5 + 1 / n * j, r_x]
                               for i in range(30) for j in range(30)], device='cuda')
            plot_Ez_fine_sample_scatter_contourf(tx, rx, n)



    # indoor_bar_pec
    TX_list = [[0, 0, 2.5], [0, 0, 1], [0, 0, 3.5], [0, -2.5, 2.5], [0, -2.5, 1], [-2.5, -2.5, 2]]
    RX_list = [0.4 + (3.8-0.4)/6 * x for x in range(7)]
    n = 49
    for t_x in TX_list:
        for r_x in RX_list:
            tx = torch.tensor([t_x] * 30 ** 2, device='cuda')
            rx = torch.tensor([[-5.5 + 11 / n * i, -5 + 10 / n * j, r_x]
                               for i in range(50) for j in range(50)], device='cuda')
            plot_Ez_fine_sample_scatter_contourf(tx, rx, n)

    # pazhou_1
    TX_list = [[369, -17, 63], [473, 16, 63], [248, 55, 63], [365, 138, 63]]
    TX_list = [[369, -17, 63]]
    RX_list = [30]
    n = 99
    for t_x in TX_list:
        for r_x in RX_list:
            tx = torch.tensor([t_x] * 1000, device='cuda')
            rx = torch.tensor([[0 + 12 * i, 0 + 25 * j, r_x]
                               for i in range(50) for j in range(20)], device='cuda')
            plot_Ez_fine_sample_scatter_contourf(tx, rx, n)



    # data_Indoor_3G_bar_e3_3cm  50*50*7
    TX_list = [[0, 0, 2], [0, 0, 3], [0, -2.5, 1], [0, -2.5, 2.5], [2.5, -2.5, 2.5], [-2.5, -2.5, 2]]
    RX_list = [0.3 + (4 - 0.3) / 6 * x for x in range(7)]
    n = 49
    for t_x in TX_list:
        for r_x in RX_list:
            tx = torch.tensor([t_x] * 2500, device='cuda')
            rx = torch.tensor([[-4.8 + 10 / n * i, -4.7 + 9.2 / n * j, r_x]
                               for i in range(50) for j in range(50)], device='cuda')
            plot_Ez_fine_sample_scatter_contourf(tx, rx, n)
    # X = -4.8: 5.2
    # Y = -4.7: 4.5
    # Z = 0.3: 4
