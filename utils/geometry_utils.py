import numpy as np
from scipy.spatial import cKDTree
# import faiss
import torch


def find_near_pcd_param_cuda(scatter_xyz, pcd_xyz, pcd_param):
    # 1. 使用 cdist 计算 scatter_xyz 和 pcd_xyz 间的成对距离
    # scatter_xyz 的形状为 (64, 128, 3)，pcd_xyz 的形状为 (5000, 3)
    # 计算结果的形状为 (64, 128, 5000)，表示每个 batch 中每个查询点到 5000 个点的距离
    distances = torch.cdist(scatter_xyz, pcd_xyz.to('cuda', dtype=torch.float32))  # 形状 (64, 128, 5000)

    # 2. 找到每个查询点最近的 pcd_xyz 索引
    # 使用 torch.min 来找到距离矩阵中最小距离的索引
    _, indices = distances.min(dim=2)  # 形状 (64, 128)

    # 3. 根据最近邻索引从 pcd_param 中提取对应的参数
    indices = indices.unsqueeze(-1)  # 形状 (64, 128, xxx)
    pcd_param = pcd_param.to('cuda', dtype=torch.float32).unsqueeze(1).expand(-1, 128, -1, -1)

    return pcd_param[indices]


def find_near_pcd_param(scatter_xyz, pcd_xyz, pcd_param):
    # 确保所有数据都在 GPU 上并使用 float32
    scatter_xyz = scatter_xyz.to('cuda', dtype=torch.float32)
    pcd_xyz = pcd_xyz.to('cuda', dtype=torch.float32)
    pcd_param = pcd_param.to('cuda', dtype=torch.float32)

    # 创建存储最近邻参数的张量
    nearest_params = torch.zeros(scatter_xyz.size(0), scatter_xyz.size(1), pcd_param.size(-1), device='cuda')

    batch_size = 16
    batch_size = min(pcd_param.size(0), batch_size)
    # batch_size = scatter_xyz.shape[0]
    # 分批处理 scatter_xyz，避免内存溢出
    for i in range(0, scatter_xyz.size(0), batch_size):
        batch_scatter = scatter_xyz[i:i + batch_size]  # 取出当前批次

        # 计算当前批次的成对距离 (batch_size, 128, 5000)
        distances = torch.cdist(batch_scatter, pcd_xyz[i:i + batch_size])

        # 找到每个点的最近邻索引
        _, indices = distances.min(dim=2)  # 形状 (batch_size, 128)

        # 提取最近邻的 pcd_param
        indices = indices.unsqueeze(-1).expand(-1, -1, pcd_param.size(-1))  # 扩展索引维度
        batch_nearest_params = torch.gather(pcd_param[i:i + batch_size].expand(-1, -1, pcd_param.size(-1)), 1, indices)

        # 存入结果
        nearest_params[i:i + batch_size] = batch_nearest_params

    return nearest_params



def find_near_pcd_param_kdtree(scatter_xyz, pcd_xyz, pcd_param):
# 返回scatter最近的pcd点云index的 pcd param

    # 使用cKDTree加速最近邻查找
    tree = cKDTree(pcd_xyz)

    # 2. 对 scatter_xyz 的每个 batch 进行查询
    batch_size = scatter_xyz.shape[0]
    num_queries = scatter_xyz.shape[1]
    nearest_params = np.empty((batch_size, num_queries, pcd_param.shape[1]), dtype=np.float32)


    for i in range(batch_size):
        # 获取当前 batch 的查询点
        batch_queries = scatter_xyz[i]  # 形状为 (128, 3)

        # 查找每个查询点在 pcd_xyz 中的最近邻
        _, indices = tree.query(batch_queries, k=1)  # 查找每个查询点最近的 1 个点，返回索引

        # 根据最近邻索引在 pcd_param 中获取参数
        nearest_params[i] = pcd_param[indices]

    # 3. 将结果转换为 torch.Tensor（如果需要）
    nearest_params_tensor = torch.from_numpy(nearest_params)

    return nearest_params_tensor

# def find_near_pcd_param_faiss(scatter_xyz, pcd_xyz, pcd_param):
#     # 1. 创建 faiss 索引
#     index = faiss.IndexFlatL2(pcd_xyz.shape[1])  # 使用 L2 距离
#     index.add(pcd_xyz)  # 将 pcd_xyz 添加到索引
#
#     # 2. 对 scatter_xyz 的每个 batch 进行查询
#     batch_size = scatter_xyz.shape[0]
#     num_queries = scatter_xyz.shape[1]
#     nearest_params = np.empty((batch_size, num_queries, pcd_param.shape[1]), dtype=np.float32)
#
#     for i in range(batch_size):
#         # 获取当前 batch 的查询点
#         batch_queries = scatter_xyz[i]  # 形状为 (128, 3)
#
#         # 查找每个查询点在 pcd_xyz 中的最近邻
#         _, indices = index.search(batch_queries, k=1)  # 查找每个查询点最近的 1 个点，返回索引
#
#         # 根据最近邻索引在 pcd_param 中获取参数
#         nearest_params[i] = pcd_param[indices.flatten()]
#
#     # 3. 将结果转换为 torch.Tensor（如果需要）
#     nearest_params_tensor = torch.from_numpy(nearest_params)
#     return nearest_params_tensor

def world_to_local_coords(l_world, v_world, n_world):
    # 归一化法向量
    n_world = n_world / torch.norm(n_world, dim=-1, keepdim=True)

    # 构造局部坐标系的 z 轴
    z_local = n_world


    # 构造参考向量，根据每个 z_local 的具体值选择 [1, 0, 0] 或 [0, 1, 0]
    ref_vector = torch.zeros_like(z_local)  # 初始化参考向量，形状与 z_local 一致
    ref_vector[..., 1] = 1  # 默认参考向量为 [0, 1, 0]
    ref_vector[..., 1] = torch.where(torch.abs(z_local[..., 1]) > 0.9, 0, 1)  # 如果接近 [0, 1, 0] 则改用 [1, 0, 0]
    ref_vector[..., 0] = torch.where(torch.abs(z_local[..., 1]) > 0.9, 1, 0)

    x_local = torch.cross(z_local, ref_vector.expand_as(z_local))
    x_local = x_local / torch.norm(x_local, dim=-1, keepdim=True)

    # 计算局部坐标系的 y 轴
    y_local = torch.cross(z_local, x_local)
    y_local = y_local / torch.norm(y_local, dim=-1, keepdim=True)

    # 构造世界坐标系到局部坐标系的旋转矩阵
    rotation_matrix = torch.stack([x_local, y_local, z_local], dim=-1)  # Shape (64, 128, 3, 3)

    # 将入射和反射向量转换到局部坐标系
    # Unsqueeze to add a dimension for matrix multiplication: (64, 128, 3) -> (64, 128, 3, 1)
    l_world = l_world.unsqueeze(-1)  # Shape (64, 128, 3, 1)
    v_world = v_world.unsqueeze(-1)  # Shape (64, 128, 3, 1)

    # Perform batch matrix multiplication (64, 128, 3, 3) * (64, 128, 3, 1)
    l_local = torch.matmul(rotation_matrix, l_world).squeeze(-1)  # Shape (64, 128, 3)
    v_local = torch.matmul(rotation_matrix, v_world).squeeze(-1)  # Shape (64, 128, 3)

    return l_local / torch.norm(l_local, dim=-1, keepdim=True), v_local / torch.norm(v_local, dim=-1, keepdim=True)


def point_line_segments_distance_gpu(P, A, B):
    """
    计算点 P 到多个线段 AB 的垂直距离，使用GPU加速
    P: 点云坐标 (torch.Tensor), 形状为 (m, 3)
    A: 线段端点 A 坐标 (torch.Tensor), 形状为 (n, 3)
    B: 线段端点 B 坐标 (torch.Tensor), 形状为 (n, 3)
    """
    # 计算线段 AB 的方向向量 (batch, n, 3)
    AB = B - A

    # 计算点 P 到 A 的向量 (batch, m, n, 3)
    AP = P.unsqueeze(2) - A.unsqueeze(1)  # (batch, m, 1, 3) - (batch, 1, n, 3) -> (batch, m, n, 3)

    # 计算线段 AB 的长度平方 (batch, n)
    AB_norm = torch.sum(AB * AB, dim=2)  # (batch, n) 线段长度的平方

    # 计算投影系数 t (batch, m, n)
    AP_AB = torch.sum(AP * AB.unsqueeze(1), dim=3)  # (batch, m, n) 点到A的向量与AB的点积
    t = AP_AB / AB_norm.unsqueeze(1)  # (batch, m, n) 投影系数

    # 计算投影点 (batch, m, n, 3)
    closest_points = A.unsqueeze(1) + t.unsqueeze(3) * AB.unsqueeze(1)

    # 筛选投影点在有效线段范围内 (0 <= t <= 1)
    # 若 t < 0，则投影点为 A
    # 若 t > 1，则投影点为 B
    t_clamped = torch.clamp(t, 0, 1)
    closest_points = A.unsqueeze(1) + t_clamped.unsqueeze(3) * AB.unsqueeze(1)

    # 计算点 P 到最近点的距离 (batch, m, n)
    distances = torch.norm(P.unsqueeze(2) - closest_points, dim=3)

    return distances, closest_points, t


def compute_attenuation_with_kdtree_gpu(A, B, point_cloud, threshold, device='cuda'):
    """closest_points
    计算点云中点到多个线段 AB 的垂直距离，判断遮挡并计算衰减系数
    A: 线段端点 A 坐标 (n, 3)
    B: 线段端点 B 坐标 (n, 3)
    point_cloud: 点云数据 (m, 3)
    threshold: 距离阈值
    device: 指定设备 ('cuda' 或 'cpu')
    """
    # 将数据移到 GPU 上
    # point_cloud_gpu = torch.tensor(point_cloud, dtype=torch.float32, device=device)
    # A_gpu = torch.tensor(A, dtype=torch.float32, device=device)
    # B_gpu = torch.tensor(B, dtype=torch.float32, device=device)

    point_cloud_gpu = torch.tensor(point_cloud).clone().detach().to(device).float()
    A_gpu = A.clone().detach().to(device).float()
    B_gpu = B.clone().detach().to(device).float()

    # 计算点到多个线段的垂直距离和最近点
    distances_gpu, closest_points, t = point_line_segments_distance_gpu(point_cloud_gpu, A_gpu, B_gpu)

    # 只保留投影在线段范围内的点，即 0 <= t <= 1
    valid_points_mask = (t >= 0) & (t <= 1)  # 生成一个掩码，只有投影在线段范围内的点为 True

    # 使用有效投影点的距离来判断遮挡情况
    valid_distances = distances_gpu * (valid_points_mask.int() * 2 - 1)  # 对无效的投影点距离置为 0（或者其他标识）

    # 统计遮挡点的个数（只考虑有效点）
    N_obstructed = torch.sum((valid_distances < threshold) & (valid_distances > 0), axis=1)  # 统计遮挡点的个数

    # 计算衰减系数（可以选择合适的衰减模型）
    attenuation = torch.exp(-1 * N_obstructed / 5)  # 简单的线性衰减

    return attenuation.unsqueeze(2)

