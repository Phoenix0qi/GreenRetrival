import os
import logging
import logging.config
import random
import numpy as np
import torch
import logging
import math
import contextlib
import time

from pathlib import Path

import yaml

from utils.geometry_utils import world_to_local_coords

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

#check_file存在
#随机种子设定
#set_logging LOGGER
#colorstr


def check_file(file):
    file = str(file)  # convert to str()
    assert os.path.isfile(file), f'File not found: {file}'  # exists
    return file

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

def set_logging(name, log_file, level=logging.INFO):
    """To set logging to file and console simultaneously"""
    logging.basicConfig(
        # level=logging.DEBUG,  # 设置日志级别为DEBUG，这样可以记录所有级别的日志信息
        level=level,  # 设置日志级别为DEBUG，这样可以记录所有级别的日志信息
        format='%(message)s',  # 日志格式，包括时间、日志级别和消息
        filename=log_file,  # 指定日志文件名
        filemode='a'  # 设置文件模式为追加模式
    )
    l = logging.getLogger(name)

    console_handler = logging.StreamHandler()

    # 设置控制台处理器的输出格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # console_handler.setFormatter(formatter)

    # 将控制台处理器添加到logger对象中
    l.addHandler(console_handler)
    return l

# LOGGING_NAME = 'sensing'
# LOGGING_DIR = 'sensing.log'
# LOGGER = set_logging(LOGGING_NAME, LOGGING_DIR, )  # run before defining LOGGER

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def check_suffix(file='yolov5s.pt', suffix=('.pt', ), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'


# parse yaml 输出 字典格式
def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data



# 检查 是否可使用 自动混合精度；需要模型用amp inference 测试
def check_amp(model):

    return False
    # # Check PyTorch Automatic Mixed Precision (AMP) functionality. Return True on correct operation
    # from models.common import AutoShape, DetectMultiBackend
    #
    # def amp_allclose(models, im):
    #     # All close FP32 vs AMP results
    #     m = AutoShape(models, verbose=False)  # models
    #     a = m(im).xywhn[0]  # FP32 inference
    #     m.amp = True
    #     b = m(im).xywhn[0]  # AMP inference
    #     return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # close to 10% absolute tolerance
    #
    # prefix = colorstr('AMP: ')
    # device = next(models.parameters()).device  # get models device
    # if device.type in ('cpu', 'mps'):
    #     return False  # AMP only used on CUDA devices
    # f = ROOT / 'data' / 'images' / 'bus.jpg'  # image to check
    # im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if check_online() else np.ones((640, 640, 3))
    # try:
    #     assert amp_allclose(deepcopy(models), im) or amp_allclose(DetectMultiBackend('yolov5n.pt', device), im)
    #     LOGGER.info(f'{prefix}checks passed ✅')
    #     return True
    # except Exception:
    #     help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
    #     LOGGER.warning(f'{prefix}checks failed ❌, disabling Automatic Mixed Precision. See {help_url}')
    #     return False

# cos loss
def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


# savedir 递增
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        # path 文件夹 or 文件
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  # 路径不存在 break 创建新文件夹
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

# 移除优化器状态 模型裁剪 去除不必要信息 存为FP16 'optimizer', 'best_fitness', 'ema', 'updates' 去除
# 'epoch': epoch,
# 'best_acc': acc_best,
# 'models': deepcopy(de_parallel(models)).half(),  # 检查模型是否是并行状态，判断并解除并行
# 'optimizer': optimizer.state_dict(),
# 'opt': vars(opt),  # 返回对象的 __dict__ 属性 字典格式的opt
# 'date': datetime.now().isoformat()
def strip_optimizer(f='best.pt', s='', LOGGER=None):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    # 移除优化器状态 模型裁剪 去除不必要信息 存为FP16 'optimizer', 'best_fitness', 'ema', 'updates' 去除
    # 保留'epoch', 'best_acc', 'models', 'optimizer', 'opt', 'date'
    x = torch.load(f, map_location=torch.device('cpu'))

    for k in ['optimizer']:  # keys
        x[k] = None

    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")

def attempt_load(weights, device=None, model_type=None, model_opt=None):
    from models.MyModel import Model_select

    ckpt = torch.load(weights, map_location='cpu')  # load
    ckpt = ckpt['model']  # FP32 models load to GPU

    assert ckpt, f'Error: No weights found in {weights}'
    assert model_opt, f'model_opt_None is None'

    model = Model_select(model_type)(model_opt)
    model.load_state_dict(ckpt.state_dict(), strict=False)
    model = model.to(device)  # create
    model.eval()

    return model

def yaml_load(file='Indoor_5G_scaters.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file='Indoor_5G_scaters.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)

def methods(instance):
    # Get class/instance methods
    # 返回类的方法，不以__开头的且能调用的, 类方法前加 @property 不可被调用
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith('__')]

class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


# modelsz = [modelout, modelin]
def check_img_size(imgsz, modelsz):
    # Checks image size (height, width)
    # 检测图像大小(高，宽)
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        assert imgsz*imgsz == modelsz[1], "Proposed imgsz is not adapted to the models"
        return [imgsz, imgsz]
    else:
        assert imgsz[0]*imgsz[1] == modelsz[1], "Proposed imgsz is not adapted to the models"
        return [imgsz[0], imgsz[1]]

def pc_normalize(pc):
    # centroid = np.mean(pc, axis=0)
    # pc = pc - centroid
    # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    # pc = pc / m
    # return pc, centroid, m

    min_vals = np.min(pc, axis=0)
    max_vals = np.max(pc, axis=0)
    # 计算每个方向的范围
    m = max_vals - min_vals
    # 计算最大范围的中心点
    center = (min_vals + max_vals) / 2.0
    # 将点云移到以中心点为原点
    pc = pc - center
    # 对点云进行缩放，使得每个坐标轴的范围都缩放到[-1, 1]之间
    pc = pc / m  # 按最大范围进行缩放
    # 返回归一化后的点云，中心点和缩放系数
    return pc, center, m

def pc_denormalize(pc, centroid, m):
    pc = pc * m + centroid
    return pc
def LoadPCS(bs, file_path):
    pcs = np.loadtxt(file_path)
    pcs, centroid, m = pc_normalize(pcs)
    pcs = np.tile(np.expand_dims(pcs, axis=0), (bs, 1, 1))
    return pcs, centroid, m

def InitPcs(npoint, bs):
    np.random.seed(0)
    theta = np.pi * np.random.random((bs, npoint))
    phi = 2 * np.pi * np.random.random((bs, npoint))
    rho = 1.0
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    pcs = np.stack([x, y, z], axis=-1)
    return pcs

def InitPcs_random_xyz(npoint, bs):
    np.random.seed(0)
    x = np.random.random((bs, npoint)) - 0.5
    y = np.random.random((bs, npoint)) - 0.5
    z = np.random.random((bs, npoint)) - 0.5

    pcs = np.stack([x, y, z], axis=-1)
    return pcs

def InitPcs_truepcs_nscatter(n_scatters, true_pcs):
    random_numbers = np.random.choice(np.arange(true_pcs.shape[1]), size=n_scatters, replace=False)
    pcs = true_pcs[0, random_numbers, :]
    return pcs

def InitPcs_uniform(npoint, bs):
    np.random.seed(0)
    np.random.seed(0)

    # 计算每个轴上的点数
    npoints_per_axis = int(np.cbrt(npoint))  # 获取每个轴上的点数，取整

    # 如果 npoint 不是完美的立方数，进行修正
    while npoints_per_axis ** 3 < npoint:
        npoints_per_axis += 1

    # 在 [-1, 1] 范围内均匀划分点
    x = np.linspace(-0.5, 0.5, npoints_per_axis)
    y = np.linspace(-0.5, 0.5, npoints_per_axis)
    z = np.linspace(-0.5, 0.5, npoints_per_axis)

    # 使用 meshgrid 生成三维网格
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    # 获取所有网格点，并根据 npoint 随机选择点
    pcs = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    # 如果点数超过所需数量，则随机选择 npoint 个点
    if pcs.shape[0] > npoint:
        np.random.shuffle(pcs)
        pcs = pcs[:npoint]

    # 将点云复制为 bs 个批次
    pcs = np.tile(pcs, (bs, 1, 1))

    return pcs

def parse_pcd_param(file_path):
    pcs = np.loadtxt(file_path)
    return pcs

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def scatter_BSDF(vector_inc, vector_sca, pcd_nearest_param):
    # pcd_nearest_param [bz, n_scatters, 4] areas 1, normals 4
    #             vector_inc = s - r
    #             vector_sca = f - r
    # P_hv 与法向无关，对称; P_hh

    in_local, sc_local = world_to_local_coords(-1 * vector_inc, vector_sca, pcd_nearest_param[..., 1:4])

    # rectangular plane BSDF
    # Spq = (k * Lx * Ly / np.sqrt(np.pi)) * np.dot(P_pq, k_d4) * np.exp(1j * k * k_r0_dot_kd)
    # cos(theta) = z  sin(theta) = (x ** 2 + y ** 2) ** 0.5
    # cos(phi) = x / (x ** 2 + y ** 2) ** 0.5   sin(phi) = y / (x ** 2 + y ** 2) ** 0.5

    def sinc(x):
        # 使用 torch 的 sin 和除法实现 sinc 函数
        return torch.sin(x) / x

    def safe_sinc(x):
        eps = 1e-8  # 避免浮点数误差
        return torch.where(torch.abs(x) < eps, 1.0, torch.sin(x) / x)

    k = 2 * np.pi * 5e9 / 3e8
    sinc_x = safe_sinc((k * pcd_nearest_param[..., 0] ** 0.5 / 2 * (sc_local[..., 0] - in_local[..., 0])) + 1e-8 )
    sinc_y = safe_sinc((k * pcd_nearest_param[..., 0] ** 0.5 / 2 * (sc_local[..., 1] - in_local[..., 1])) + 1e-8 )
    in_local_dot_sc_local = torch.sum(in_local * sc_local, dim=-1)  # 点积计算
    # Spq = torch.abs(sinc_x * sinc_y * pcd_nearest_param[..., 0] / torch.pi ** 0.5 * in_local[..., 2] * in_local_dot_sc_local)
    Spq = torch.abs(sinc_x * sinc_y * in_local[..., 2] * in_local_dot_sc_local) # 只预测方向相关函数，幅度不考虑

    return Spq

def pattern_in_sc(vector_inc, vector_sca, rcube_sh=None, deg=0, pcd_nearest_param=None):
    #             vector_inc = s - r
    #             vector_sca = f - r

    if rcube_sh is not None:  # sh
        in_s = eval_sh(deg, rcube_sh[..., 0], vector_inc)
        sc_s = eval_sh(deg, rcube_sh[..., 1], vector_sca)
        # sc_pattern = in_s * sc_s
        sc_pattern = sc_s
    elif pcd_nearest_param is not None:  # BSDF
        sc_pattern = scatter_BSDF(vector_inc, vector_sca, pcd_nearest_param)
    else:
        sc_pattern = None

    return sc_pattern


# 生成球面上的单位方向
def generate_spherical_coords(num_points=1000):
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta = np.linspace(0, np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack([x, y, z], axis=-1)

# 可视化球谐函数结果
def plot_spherical_surface(result, dirs):
    import matplotlib.pyplot as plt
    # 转换到笛卡尔坐标系
    x = result * dirs[..., 0]
    y = result * dirs[..., 1]
    z = result * dirs[..., 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制球面
    ax.plot_surface(x, y, z, cmap='inferno', edgecolor='k')

    # 设置图形的标题和轴
    ax.set_title("Spherical Harmonics Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


# # 计算球谐函数值
# deg = 3  # 阶数
# # sh = rcube_sh[0,0,:,1].unsqueeze(0).unsqueeze(0)
# dirs = generate_spherical_coords()  # 生成球面方向
# dirs = torch.tensor(dirs, device='cuda')
# # 评估球谐函数
# result = eval_sh(deg, sh, dirs)
# plot_spherical_surface(result.cpu().detach().numpy(), dirs.cpu().detach().numpy())