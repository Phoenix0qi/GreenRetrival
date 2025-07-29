import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

def KNN_interp(train_dataloader, val_dataloader, save_dir):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    # Assume training data (T_x, R_x, E) and testing data (T_x, R_x)
    # Example: training_data and test_data are numpy arrays of shape (N, 2) for Tx, Rx and E is a vector of shape (N,)
    # T_x, R_x are in 2D arrays, where first column is T_x, second column is R_x.

    pbar = enumerate(train_dataloader)
    Ez_train_gt = []
    Tx_train = []
    Rx_train = []
    for batch_i, (inputs, labels) in pbar:
        # tloc, rloc
        inputs_train = [input.to(device) for input in inputs]  # [tloc, rloc]
        Tx_train.append(inputs_train[0])
        Rx_train.append(inputs_train[1])
        Ez_train_gt.append(labels[0])

    pbar = enumerate(val_dataloader)
    Ez_val_gt = []
    Tx_val = []
    Rx_val = []
    for batch_i, (inputs, labels) in pbar:
        # tloc, rloc
        inputs_val = [input.to(device) for input in inputs]  # [tloc, rloc]
        Tx_val.append(inputs_val[0])
        Rx_val.append(inputs_val[1])
        Ez_val_gt.append(labels[0])

    Ez_train_gt = torch.cat(Ez_train_gt, dim=0).to(torch.float32).to(device)
    Tx_train = torch.cat(Tx_train, dim=0).to(torch.float32).to(device)
    Rx_train = torch.cat(Rx_train, dim=0).to(torch.float32).to(device)

    Ez_val_gt = torch.cat(Ez_val_gt, dim=0).to(torch.float32).to(device)
    Tx_val = torch.cat(Tx_val, dim=0).to(torch.float32).to(device)
    Rx_val = torch.cat(Rx_val, dim=0).to(torch.float32).to(device)



    k = 4

    def estimate_E_for_matching_Tx(Tx_train, Rx_train, Ez_train_gt, Tx_val, Rx_val, k=4):
        estimated_E_real = []
        estimated_E_imag = []

        # Loop through each test Tx
        for test_Tx, test_Rx in zip(Tx_val, Rx_val):
            # Find the matching Tx in the training set (we assume there's a direct match)
            matching_indices = torch.where(torch.all(Tx_train - test_Tx < 1e-5, dim=1))[0]  # Find all matching Tx

            if len(matching_indices) == 0:
                # No matching Tx, we could handle this case or skip it
                continue

            # Extract the corresponding Rx values for matching Tx
            matching_Rx_train = Rx_train[matching_indices]  # All Rx corresponding to the matching Tx

            # Compute the Euclidean distances between test Rx and matching Rx in training
            diff = test_Rx.unsqueeze(0) - matching_Rx_train  # Shape (1, k, 3)
            distances = torch.norm(diff, dim=1)  # Euclidean distance, shape (k,)

            # Get the indices of the k nearest neighbors based on distances
            _, nearest_indices = torch.topk(distances, k, largest=False, sorted=False)

            # Extract the corresponding Ez values for the nearest neighbors
            nearest_Ez_values = Ez_train_gt[matching_indices[nearest_indices]]  # Get Ez for the k nearest neighbors

            # Separate real and imaginary parts of Ez
            nearest_E_real = nearest_Ez_values[:, 0]  # Real part of Ez
            nearest_E_imag = nearest_Ez_values[:, 1]  # Imaginary part of Ez

            # Average the real and imaginary parts for interpolation
            estimated_E_real.append(nearest_E_real.mean())  # Average real part
            estimated_E_imag.append(nearest_E_imag.mean())  # Average imaginary part

            print(f"{test_Tx}\t{test_Rx}")

        # # Combine real and imaginary parts into complex numbers
        estimated_E = torch.stack([torch.tensor(estimated_E_real, device=device),
                                                 torch.tensor(estimated_E_imag, device=device)]).to(device)
        return estimated_E

    # Estimate Ez for the validation data
    estimated_Ez_val = estimate_E_for_matching_Tx(Tx_train, Rx_train, Ez_train_gt, Tx_val, Rx_val, k)

    # Print estimated Ez values for the validation data
    print("Estimated Ez values for the validation data:", estimated_Ez_val)

    def plot_signal(pred, gt, title, savedir, filename=""):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(gt, label='Ground Truth')
        plt.plot(pred, label='Prediction')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(savedir, filename), dpi=300)
        plt.close()

    Ez_pred_Re = (estimated_Ez_val[0, :]).cpu()
    Ez_pred_Im = (estimated_Ez_val[1, :]).cpu()
    Ez_pred_abs = (Ez_pred_Re ** 2 + Ez_pred_Im ** 2) ** 0.5

    Ez_gt_Re = Ez_val_gt[:, 0].cpu()
    Ez_gt_Im = Ez_val_gt[:, 1].cpu()
    Ez_gt_abs = ((Ez_val_gt[:, 0] ** 2 + Ez_val_gt[:, 1] ** 2 ) **0.5).cpu()

    plot_signal(Ez_pred_Re, Ez_gt_Re, "Re_Ez", save_dir, "Re_Ez")
    plot_signal(Ez_pred_Im, Ez_gt_Im, "Im_Ez", save_dir, "Im_Ez")
    plot_signal(Ez_pred_abs, Ez_gt_abs, "abs_Ez", save_dir, "abs_Ez")

    plot_signal(Ez_pred_Re[0:100], Ez_gt_Re[0:100], "Re_Ez", save_dir, "Re_Ez_100")
    plot_signal(Ez_pred_Im[0:100], Ez_gt_Im[0:100], "Im_Ez", save_dir, "Im_Ez_100")
    plot_signal(Ez_pred_abs[0:100], Ez_gt_abs[0:100], "abs_Ez", save_dir, "abs_Ez_100")


    with open(os.path.join(save_dir, "result_val.txt"), 'a') as f:
        for i in range(len(Ez_gt_Re)):
            f.write("{:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(Ez_gt_Re[i].item(), Ez_pred_Re[i].item(), Ez_gt_Im[i].item(), Ez_pred_Im[i].item()))



    # with open(os.path.join(save_dir, "result_val.txt"), 'a') as f:
    #     for (rssi, gt_rssi) in zip(Ez_val_gt, estimated_Ez_val):  # 写反了
    #         f.write("{:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(rssi[0],
    #                                                           gt_rssi[0],
    #                                                           rssi[1],
    #                                                           gt_rssi[1]
    #                                                           ))

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

    # # indoor 3cm
    # TX_list = [[0, 0, 2], [0, 0, 3], [0, -2.5, 1], [0, -2.5, 2.5], [2.5, -2.5, 2.5], [-2.5, -2.5, 2]]
    # RX_list = [0.3 + (4 - 0.3) / 6 * x for x in range(7)]
    # n = 49
    # for t_x in TX_list:
    #     for r_x in RX_list:
    #         tx = torch.tensor([t_x] * 2500, device='cuda')
    #         rx = torch.tensor([[-4.8 + 10 / n * i, -4.7 + 9.2 / n * j, r_x]
    #                            for i in range(50) for j in range(50)], device='cuda')
    #         # plot_Ez_fine_sample_scatter_contourf(tx, rx, n)

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


    # sinna_PKU
    # TX_list = [[0, 0, 40], [0, -300, 40]]
    TX_list = [[0, 0, 40]]
    RX_list = [1]
    n = 76800
    n_x = 240 - 1
    n_y = 320 - 1
    for t_x in TX_list:
        for r_x in RX_list:
            tx = torch.tensor([t_x] * (n_x + 1) * (n_y + 1), device='cuda')
            rx = torch.tensor([[-600 + 1200 / n_x * i, -800 + 1600 / n_y * j, r_x]
                      for i in range(n_x + 1) for j in range(n_y + 1)], device='cuda')




            # bz = 100  # 300  # 96
            # for i in range(int(tx.shape[0] / bz)):
            #     # block_pcd_val = np.tile(np.expand_dims(block_pcd, axis=0), (bz, 1, 1))
            #     # pred = model(tx[i*bz: (i+1)*bz, :], rx[i*bz: (i+1)*bz, :], if_Ez_theta, if_s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd_val) # block
            #
            #     with torch.no_grad():
            #         pred = estimate_E_for_matching_Tx(Tx_train, Rx_train, Ez_train_gt, tx[i * bz: (i + 1) * bz, :], rx[i * bz: (i + 1) * bz, :], k=4)
            #
            #     pred = [pred.cpu().detach() for pred in pred]
            #
            #     # Ez_pred_Re = (estimated_Ez_val[0, :]).cpu()
            #     # Ez_pred_Im = (estimated_Ez_val[1, :]).cpu()
            #     # Ez_pred_abs = (Ez_pred_Re ** 2 + Ez_pred_Im ** 2) **0.5
            #
            #     color_pred = torch.sqrt(pred[0][:, 0] ** 2 + pred[0][:, 1] ** 2)
            #     color_pred_Re = pred[0][:, 0]
            #     color_pred_Im = pred[0][:, 1]
            #     color_pred = torch.nan_to_num(color_pred)
            #     # color_pred = color_pred.cpu().detach().numpy()
            #     color.append(color_pred)
            #     color_Re.append(color_pred_Re)
            #     color_Im.append(color_pred_Im)

            with torch.no_grad():
                pred = estimate_E_for_matching_Tx(Tx_train, Rx_train, Ez_train_gt, tx, rx, k=4)

            # pred = [pred.cpu().detach() for pred in pred]


            Ez_pred_Re = (pred[0, :]).cpu()
            Ez_pred_Im = (pred[1, :]).cpu()
            Ez_pred_abs = (Ez_pred_Re ** 2 + Ez_pred_Im ** 2) **0.5

            color = np.array(Ez_pred_abs).reshape(-1)
            color_Re = np.array(Ez_pred_Re).reshape(-1)
            color_Im = np.array(Ez_pred_Im).reshape(-1)

            # # plt.scatter(rx[:, 0].cpu(), rx[:, 1].cpu(), c=color, s=1) # sionna s=1
            # plt.scatter(rx[:, 0].cpu(), rx[:, 1].cpu(), c=color, s=30)  # indoor bar 3cm s=30
            # plt.colorbar(label="Color Intensity")  # 添加颜色条
            # plt.savefig(os.path.join(save_dir, f"TX_{t_x}_points_{n}_scatter_RX_xyplane_z_{rx[-1, -1]:.3f}.png"), dpi=300)
            # # plt.show()
            # plt.close()


            epoch = 0

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
            # plt.scatter(x, y, c=amplitude, s=30)  # indoor bar s=30
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
            # plt.scatter(x, y, c=amplitude, s=30)  # indoor bar s=30
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
            # plt.scatter(x, y, c=amplitude, s=30)  # indoor bar s=30
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
