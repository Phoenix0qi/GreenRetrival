
import torch
import os
import matplotlib.pyplot as plt


result_path_total = r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\data_results"

def plot_lines(result_path):
    with open(result_path, 'r') as f:
        lines = f.readlines()
    f.close()

    save_path = os.path.dirname(result_path)
    gt, pred = [], []
    for i, line in enumerate(lines):
        values = line.split(',')
        gt.append([float(values[0]), float(values[2])])
        pred.append([float(values[1]), float(values[3])])

    labels = torch.tensor(gt)
    outputs = torch.tensor(pred)

    # A4纸宽度：210mm = 8.27英寸，取1/5宽度
    fig_width = 8.27 / 5 * 1.3  # 1.654 inch
    fig_height = fig_width * 6 / 10   # 保持10:6的比例

    plt.rcParams.update({
        'font.size': 9,  # 8号字 ≈ 10.5 pt
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.titlesize': 9,
    })
    plt.figure(figsize=(fig_width, fig_height))  # 单位是英寸


    # === 开始画图 ===
    plt.plot(labels[0:100, 0], label='Ground Truth', linewidth=1.0)
    plt.plot(outputs[0:100, 0], label='Prediction', linewidth=1.0)
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.legend()
    # === 保存图片 ===
    plt.savefig(os.path.join(save_path, "Re_hard_100.png"), dpi=300, bbox_inches='tight')  # bbox_inches='tight'让周围留白很少
    plt.savefig(os.path.join(result_path_total, save_path.split("\\")[-1] + "Re_hard_100.png"), dpi=300, bbox_inches='tight')
    plt.close()


    # A4纸宽度：210mm = 8.27英寸，取1/5宽度
    fig_width = 8.27 / 5 * 1.3  # 1.654 inch
    fig_height = fig_width * 6 / 10   # 保持10:6的比例

    plt.rcParams.update({
        'font.size': 9,  # 8号字 ≈ 10.5 pt
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.titlesize': 9,
    })
    plt.figure(figsize=(fig_width, fig_height))  # 单位是英寸

    # === 开始画图 ===
    plt.plot(labels[0:100, 1], label='Ground Truth', linewidth=1.0)
    plt.plot(outputs[0:100, 1], label='Prediction', linewidth=1.0)
    plt.savefig(os.path.join(save_path, "Im_hard_100.png"), dpi=300, bbox_inches='tight')  # bbox_inches='tight'让周围留白很少
    plt.savefig(os.path.join(result_path_total, save_path.split("\\")[-1] + "Im_hard_100.png"), dpi=300, bbox_inches='tight')
    plt.close()





def plot_NeRF2_lines(result_NeRF_path):
    with open(result_NeRF_path, 'r') as f:
        lines = f.readlines()
    f.close()

    save_path = os.path.dirname(result_NeRF_path)
    gt, pred = [], []
    for i, line in enumerate(lines):
        values = line.split(',')
        gt.append(float(values[0])*100) # / 1000
        pred.append(float(values[1])*100) # *100 pazhou
        # if i == 100: break

    # A4纸宽度：210mm = 8.27英寸，取1/5宽度
    fig_width = 8.27 / 5 * 1.3  # 1.654 inch
    fig_height = fig_width * 6 / 10   # 保持10:6的比例

    plt.rcParams.update({
        'font.size': 9,  # 8号字 ≈ 10.5 pt
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.titlesize': 9,
    })
    plt.figure(figsize=(fig_width, fig_height))  # 单位是英寸

    # === 开始画图 ===
    plt.plot(gt[0:100], label='Ground Truth', linewidth=1.0)
    plt.plot(pred[0:100], label='Prediction', linewidth=1.0)
    plt.savefig(os.path.join(save_path, "Re_hard_100.png"), dpi=300, bbox_inches='tight')  # bbox_inches='tight'让周围留白很少
    plt.savefig(os.path.join(result_path_total, save_path.split("\\")[-1] + "Re_hard_100.png"), dpi=300, bbox_inches='tight')
    plt.close()



    gt, pred = [], []
    for i, line in enumerate(lines):
        values = line.split(',')
        gt.append(float(values[2])*100)
        pred.append(float(values[3])*100)
        # if i == 100: break

    # A4纸宽度：210mm = 8.27英寸，取1/5宽度
    fig_width = 8.27 / 5 * 1.3  # 1.654 inch
    fig_height = fig_width * 6 / 10  # 保持10:6的比例

    plt.rcParams.update({
        'font.size': 9,  # 8号字 ≈ 10.5 pt
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.titlesize': 9,
    })
    plt.figure(figsize=(fig_width, fig_height))  # 单位是英寸

    # === 开始画图 ===
    plt.plot(gt[0:100], label='Ground Truth', linewidth=1.0)
    plt.plot(pred[0:100], label='Prediction', linewidth=1.0)
    plt.savefig(os.path.join(save_path, "Im_hard_100.png"), dpi=300, bbox_inches='tight')  # bbox_inches='tight'让周围留白很少
    plt.savefig(os.path.join(result_path_total, save_path.split("\\")[-1] + "Im_hard_100.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

# result_path_each = [r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\cube_fine_2\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\Cube_3G_e5_sample_fine_KNN4\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\Cube_3G_e5_sample_fine_MLP_test_plot\result_val.txt",

#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\indoor_3G_3cm_mask_txt_MSE_loss_random_test_last_scatter_2\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\indoor_e3_3cm_dataset_KNN4\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\indoor_e3_3cm_dataset_normPosEnc_MLP_train_2000_2_best_test_plot\result_val.txt",

#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\pazhou_4TX_mask_12_last_txt_scatter\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_pazhou_34e8_scatters_KNN4\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\pazhou_34e8_scatters_result2_4TX_posNormEnc_MLP_2_test_plot\result_val.txt",

#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\cube_same_NeRF_720_64_test_best\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\cube_same_NeRF_KNN4_3\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\cube_3G_sameas_NeRF2_posNormEnc_MLP\result_val.txt",]

# result_path_each = [r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_indoor_2_4G_VV_lr0001_bz_64\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_indoor_3_4G_VV_lr0001_bz_64\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_indoor_2_4G_VV_3\result_val.txt",
#                     r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_indoor_3_4G_VV_3\result_val.txt"] 

result_path_each = [r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_indoor_3G_HV_lr0001_bz_64_test\result_val.txt",
                    r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_indoor_3G_HH_lr0001_bz_64_test\result_val.txt",
                    r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_indoor_3G_VH_lr0001_bz_64_test\result_val.txt"] 

for item in result_path_each:
    plot_lines(item)

result_NeRF2_path = [
                     r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\Cube_3G_e5_sample_fine\Cube_3G_e5_sample_fine_200k_iter_x1\result.txt",
                     r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\indoor_e3_3cm_dataset_3G\indoor_e3_3cm_dataset_3G\result.txt",
                     # r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\pazhou_34e8_scatters_result2_4TX\pazhou_34e8_scatters_result2_4TX\result.txt",]
                     r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\single_cube_ReIm\cube_3G_702_64_ReIm_retrain_200k_iter\result.txt"]

# for item in result_NeRF2_path:
#     plot_NeRF2_lines(item)



result_fig_path = {
    "Cube_3G_fine_GT"       : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\datasets\Cube_3G_e5_sample_fine\gt_fine_Ez",
    "Ours"                  : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian-Green\runs\train\DATA_data_Cube_3G_e5_sample_fine_x1_MODEL_Green_simple_v2_Green_all_1000_LOSS_chamfer_01\fine_sample_Ez_29",
    "NeRF"                  : r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\Cube_3G_e5_sample_fine\Cube_3G_e5_sample_fine_200k_iter_x1",
    "KNN"                   : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\Cube_3G_e5_sample_fine_KNN4_plot_test_0.15",
    "MLP"                   : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\Cube_3G_e5_sample_fine_MLP_test_plot" ,
    "MLP2"                  : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\Cube_3G_e5_sample_fine_MLP_noNorm_test_plot",

    "Indoor_e5_3cm_GT"      : r"D:\ResearchWorkCodes\FEKO_test\FEKO_INDOOR_E5\indoor_e3_3cm_gt",
    "Ours"                  : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\indoor_3G_3cm_mask_txt_MSE_loss_random_test_last_scatter_2",
    "NeRF"                  : r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\indoor_e3_3cm_dataset_3G\indoor_e3_3cm_dataset_3G",
    "KNN"                   : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\indoor_e3_3cm_dataset_KNN4_test_plot",
    "MLP"                   : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\indoor_e3_3cm_dataset_normPosEnc_MLP_train_2000_2_best_test_plot",


    "Pazhou_GT"             : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\datasets\pazhou_34e8_scatters_result2_4TX\gt",
    "Ours"                  : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\pazhou_4TX_mask_12_last_txt_scatter",
    "NeRF"                  : r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\pazhou_34e8_scatters_result2_4TX\pazhou_34e8_scatters_result2_4TX",
    "KNN"                   : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\data_pazhou_34e8_scatters_KNN4_test_plot",
    "MLP"                   : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\pazhou_34e8_scatters_result2_4TX_posNormEnc_MLP_2_test_plot",


    "Ours *1"               : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian-Green\runs\train\DATA_data_Cube_3G_e5_sample_fine_x1_MODEL_Green_simple_v2_Green_all_1000_LOSS_chamfer_01\fine_sample_Ez_29",
    "NeRF *1"               : r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\Cube_3G_e5_sample_fine\Cube_3G_e5_sample_fine_200k_iter_x1\result_1_29",
    "Ours *0.5"             : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian-Green\runs\train\DATA_data_Cube_3G_e5_sample_fine_x05_MODEL_Green_simple_v2_Green_all_1000_LOSS_chamfer_01\fine_sample_Ez_29",
    "NeRF *0.5"             : r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\Cube_3G_e5_sample_fine\Cube_3G_e5_sample_fine_200k_iter_x05",
    "Ours *0.1"             : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian-Green\runs\train\DATA_data_Cube_3G_e5_sample_fine_x01_MODEL_Green_simple_v2_Green_all_1000_LOSS_chamfer_01\fine_sample_Ez_29",
    "NeRF *0.1"             : r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\Cube_3G_e5_sample_fine\Cube_3G_e5_sample_fine_200k_iter_x01",


    "Ours_cube_sameas_NeRF" : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian-Green\runs\train\test_DATA_data_cube_3G_sameas_NeRF2_MODEL_Green_simple_v2_Green_all_1000_LOSS_chamfer_01\fine_sample_Ez",
    "NeRF_cube_sameas_NeRF" : r"D:\ResearchWorkCodes\Pyworkspace20240211\NeRF2\logs\single_cube_ReIm\cube_3G_702_64_ReIm_retrain_200k_iter",
    "KNN_same"              : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\cube_3G_sameas_NeRF2_KNN4_test_plot",
    "MLP_same"              : r"D:\ResearchWorkCodes\Pyworkspace20240211\Gaussian_Green_point_opti\runs\train\cube_3G_sameas_NeRF2_posNormEnc_MLP_test_plot",
}