import os
import torch
import sys
from tqdm import tqdm
import argparse
import time
from pathlib import Path
import logging
import yaml
from copy import deepcopy
from shutil import copyfile
from datetime import datetime

from torch.optim import lr_scheduler

from utils.general import (increment_path, set_logging, colorstr, yaml_save, init_seeds, methods, check_suffix, check_amp,
                           TQDM_BAR_FORMAT, one_cycle, strip_optimizer, attempt_load, LoadPCS, parse_yaml, pc_denormalize,
                           parse_pcd_param, InitPcs, InitPcs_uniform, InitPcs_random_xyz, InitPcs_truepcs_nscatter)
from utils.callbacks import Callbacks
from utils.loggers.loggers import Loggers
from utils.torch_utils import smart_optimizer, de_parallel
from utils.loss import ComputeLoss

from models.MyModel import Model_select
from data_loaders.MyDataloader import create_dataloader
import val as validate

from metrics.NMSE import compute_NMSE, compute_NMSE_Ez_j

import numpy as np

from KNN_inter_functions import KNN_interp


import time


FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径 train.py main.py
ROOT = FILE.parents[0]  # YOLOv5 root directory ROOT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def training(hyp, opt, device, callbacks, LOGGER):
    weights, data, batch_size, workers, epochs, nosave, noval, save_dir, freq, n_scatters, pcd_file_name, plot_epoch, reverse_conj, end_train_plot, plot_scatters, Ez_theta, s_pattern, scatter_BSDF, if_sh_module, deg, test_model_path, block, point_opti, KNN_mode = \
        opt.weights, opt.data, opt.batch_size, opt.workers, opt.epochs, opt.nosave, opt.noval, opt.save_dir, opt.freq, opt.n_scatters, opt.pcd_file_name, opt.plot_epoch, opt.reverse_conj, opt.end_train_plot, opt.plot_scatters, opt.Ez_theta, opt.s_pattern, opt.scatter_BSDF, opt.if_sh_module, opt.deg, opt.test_model_path, opt.block, opt.point_opti, opt.KNN_mode

    # Directories savedir
    save_dir = Path(save_dir)
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str) or isinstance(hyp, Path):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # seeds
    init_seeds(opt.seed, deterministic=True)

    # Loggers
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
    # Register actions
    for k in methods(loggers): # 把Logger中所有方法注册，使用callbacks.run('xxxx')执行
        callbacks.register_action(k, callback=getattr(loggers, k))

    # Model & Pretrained
    check_suffix(weights, '.pt')  # check weights  未写
    pretrained = str(weights).endswith('.pt') # 是否pretrained

    data_yaml = parse_yaml(data)
    data_rootdir = data_yaml['data_path']
    pcd_file_path = os.path.join(data_rootdir, pcd_file_name)
    # load scatters, gt norm
    true_pcs, centroid, m = LoadPCS(batch_size, pcd_file_path)

    Model = Model_select(hyp['Model_type'])([centroid, m, batch_size, freq, n_scatters])
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model.to(device)  # create
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'].state_dict(), strict=False)  # load 存储的模型为带其他参数的model
        else:
            model.load_state_dict(ckpt.state_dict(), strict=False)
        LOGGER.info(f'Pretrain load model:{weights}')  # report
    else:
        model = Model.to(device)  # create

    # 是否自动混合精度
    amp = check_amp(model)  # check AMP  return False

    # Freeze

    if pretrained:
        init_pc = ckpt['rcube_']
    else:
        init_pc = InitPcs_truepcs_nscatter(n_scatters, true_pcs)
        # init_pc = InitPcs_random_xyz(n_scatters, 1)

    rcube_ = torch.tensor(init_pc, requires_grad=False, dtype=torch.float32, device=device)

    if pretrained:
        ckai_ = ckpt['ckai_']
    else:
        ckai_ = torch.rand(n_scatters, 2)  # rand zero

    ckai_ = torch.tensor(ckai_, requires_grad=True, dtype=torch.float32, device=device)
    # ckai_ = torch.tensor(torch.ones(n_scatters, 1), requires_grad=True, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW([  # Adam DGS
        {'params': rcube_, 'lr': hyp['lr0']*0.1},  # 为 rcube_ 设置学习率  hyp['lr0']
        {'params': ckai_, 'lr': hyp['lr0']*0.1},  # 为 ckai_ 设置不同的学习率
        {'params': model.parameters(), 'lr': hyp['lr0']*0.01}
    ], weight_decay=5e-3) #  momentum=0.9,

    accumulate = 0 # 不使用梯度累积

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine: 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear: 1->hyp['lrf']

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)


    start_epoch = 0


    # Trainloader
    # 传入data yaml; batch size; aug; workers; seed; shuffle
    # return train_loador, dataset类
    # 处理数据集操作 aug等
    train_loader, dataset = create_dataloader(data,     # yaml文件
                                              trainorval="train",
                                              batch_size=batch_size,
                                              # hyp=hyp, #是否数据处理
                                              augment=False,
                                              workers=workers,
                                              shuffle=True,
                                              seed=opt.seed,
                                              LOGGER=LOGGER)


    val_loader, _ = create_dataloader(data,
                                      trainorval="val",
                                      batch_size=batch_size,
                                      # hyp=hyp, #是否数据处理
                                      augment=False,
                                      workers=workers,
                                      shuffle=False,
                                      seed=opt.seed,
                                      LOGGER=LOGGER)

    # callbacks
    callbacks.run('on_pretrain_routine_end')



    # Model attributes 设置模型变化参数
    # model hyp
    model.hyp = hyp  # attach hyperparameters to model
    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    # warmup
    nw = max(round(hyp['warmup_epochs'] * nb), 0)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # optimize 梯度裁剪
    last_opt_step = -1
    # 定义results
    mse_best = -1.0
    val_loss = 0.0
    # 将 scheduler 的 last_epoch 设置为 start_epoch - 1 确保在训练开始之前不会影响学习率的调整；初始化为 -1，表示尚未进行任何训练；每次调用 scheduler.step() 后，last_epoch 的值都会递增，以记录训练的进度。
    scheduler.last_epoch = start_epoch - 1
    # # amp loss scaler
    scaler = torch.amp.GradScaler(enabled=amp) # loss scaler
    # early stop
    # stopper, stop = EarlyStopping(patience=opt.patience), False

    # compute loss
    compute_loss = ComputeLoss(model.hyp)  # init loss class

    callbacks.run('on_train_start')
    LOGGER.info(f'Starting training for {epochs} epochs...')
    # callback 重写

    if KNN_mode:
        KNN_interp(train_loader, val_loader, save_dir)

    elif not str(test_model_path).endswith('.pt'):

        infer_time_mean = 0.0
        end_time_mean = 0.0

        # ======================= epoch ============================
        for epoch in range(start_epoch, epochs):
            callbacks.run('on_train_epoch_start')
            model.train()

            pbar = enumerate(train_loader)
            # Logger
            LOGGER.info(('\n' + '%11s' * 6) % ('Epoch', 'GPU_mem', 'l_total', 'l_Ez', 'l_chamfer', 'l_scatter'))
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

            optimizer.zero_grad()  # 清理梯度缓存
            # 定义loss accuracy
            # 根据 数据形式改 1个epoch的mloss
            # mloss = torch.tensor([0.0, 0.0, 0.0]).to(device) # mean losses: total, Ez, chamfer, theta
            mloss = [0.0, 0.0, 0.0, 0.0]
            NMSE = 0.0

            # ======================= batch ============================
            #  改 根据数据类型dataloader中格式 改
            for i, (inputs, labels) in pbar:


                start_time = time.time()

                callbacks.run('on_train_batch_start')

                ni = i + nb * epoch  # number integrated batches (since train start)

                # 输入数据 inputs
                # inputs: tloc, rloc
                inputs = [input.to(device) for input in inputs] # [tloc, rloc]
                # labels [gtot, scat_gt]

                ### input : [tloc, rloc]
                ### model.forward(self, tloc, rloc))
                ### output: [gtot, rcube_, rcube, ckai] if Ez_theta:  [gtot, [rcube_, gtot_scatter], rcube, ckai]

                # Forward
                with torch.amp.autocast('cuda', enabled=amp):  # 混合精度运算

                    if scatter_BSDF:
                        pcd_param_path = os.path.join(data_rootdir, scatter_BSDF)
                        pcd_param = parse_pcd_param(pcd_param_path)  # [n_points, 4]  [pcd_areas 1, pcd_normals 3]
                        # pcd_param = torch.tensor(pcd_param)
                        pcd_param = torch.tile(torch.tensor(pcd_param), (inputs[0].shape[0], 1, 1))
                        pcd_xyz = torch.tensor(pc_denormalize(true_pcs, centroid, m))[0:inputs[0].shape[0]+1, ...]
                        pred = model(inputs[0], inputs[1], Ez_theta, s_pattern, pcd=[pcd_xyz, pcd_param])  # [pcd_xyz, pcd_areas, pcd_normals]
                    elif block:
                        block_pcd = np.loadtxt(pcd_file_path)
                        block_pcd = np.tile(np.expand_dims(block_pcd, axis=0), (min(batch_size, inputs[0].shape[0]), 1, 1))
                        # pred = model(inputs[0], inputs[1], Ez_theta, s_pattern, sh_module=if_sh_module, deg=deg,
                        #              block_pcd=block_pcd)
                        if point_opti:
                            pred = model(inputs[0], inputs[1], Ez_theta, s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd, rcube_=rcube_, ckai_=ckai_)
                        else:
                            pred = model(inputs[0], inputs[1], Ez_theta, s_pattern, sh_module=if_sh_module, deg=deg,
                                         block_pcd=block_pcd)

                    else:  # Sh module
                        pred = model(inputs[0], inputs[1], Ez_theta, s_pattern, sh_module=if_sh_module, deg=deg)

                    if true_pcs.shape[0] != len(inputs[0]):
                        true_pcs_label = true_pcs[0:len(inputs[0]), ...]
                    else:
                        true_pcs_label = true_pcs

                    # 计算 loss  loss类型  # compute_loss 类 写具体loss  compute_loss(outputs, labels)
                    # outputs: [Ez_ReIm, scatters_xyz]
                    # labels: [Ez_gt_ReIm, true_pcd_xyz, [scatters_gt_xyz, rloc]
                    labels = [label.to(device) for label in labels]  # [gtot, scat_gt]
                    loss = compute_loss(pred[0:2], [labels[0], torch.tensor(true_pcs_label, dtype=torch.float32).to(device), [labels[1], inputs[1]]])  # loss scaled by batch_size ？？
                    # loss: [loss, loss_Ez, loss_rp, orient_cos_loss]
                    loss_sum = loss[0].sum()


                infer_time = time.time()

                # Backward
                # 使用amp 缩放损失值，并在反向传播后执行梯度裁剪、参数更新和梯度清零
                # 使用梯度缩放器进行缩放
                scaler.scale(loss_sum).backward()


                end_time = time.time()

                infer_time_mean += infer_time - start_time
                end_time_mean += end_time - start_time

                # print(f"\n infer_time: {infer_time_mean / (ni+1):.4f} s; total_time: {end_time_mean / (ni+1):.4f} s \n")

                if epoch == 0 and i == 1:
                    # print(f"\n\n{i}\n\n")
                    # 设置 profiler
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_flops=True,
                        profile_memory=True,
                        with_stack=False  # 可设为True用于分析源码调用路径
                    ) as prof:
                        with torch.no_grad():  # 如果你只分析推理
                            recv_signal = model(inputs[0], inputs[1], Ez_theta, s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd, rcube_=rcube_, ckai_=ckai_)

                    # 打印按 FLOPs 排序的前20个操作
                    print(prof.key_averages().table(sort_by="flops", row_limit=20))

                    # 如果你要分析整个函数的总 FLOPs：
                    total_flops = sum([item.flops for item in prof.key_averages() if item.flops is not None])
                    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")

                    # 参数量
                    used_params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                    total_params_used = sum(p.numel() for p in used_params)
                    print(f"Params used in current backward: {total_params_used}")

                    extra_params = [rcube_, ckai_]
                    total_params_extra = sum(p.numel() for p in extra_params if p.requires_grad)
                    print(f"Extra params total: {total_params_extra}")

                    total_params_all = total_params_used + total_params_extra
                    
                    def to_MB(p): return p * 4 / (1024 ** 2)
                    print(f"Model used params: {total_params_used:,} ({to_MB(total_params_used):.2f} MB)")
                    print(f"Extra params     : {total_params_extra:,} ({to_MB(total_params_extra):.2f} MB)")
                    print(f"Total            : {total_params_all:,} ({to_MB(total_params_all):.2f} MB)")


                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= accumulate: # 在warmup batch之后 进行梯度裁剪
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients  #设置 threshold
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    last_opt_step = ni


                # Log [loss, loss_rp, orient_cos_loss]
                mloss = [(mloss_item * i + loss_item) / (i + 1) for mloss_item, loss_item in zip(mloss, loss)]# update mean losses  # 计算一个epoch内动态变化 (乘i + 当前loss)/当前总batch
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # GPU memory (GB) (GB)

                NMSE = (NMSE * i + compute_NMSE(labels[0], pred[0])) / (i + 1)

                pbar.set_description(('%11s' * 2 + '%11.4g' * 4) %
                                     (f'{epoch}/{epochs - 1}', mem, mloss[0], mloss[1], mloss[2], mloss[3]))

                callbacks.run('on_train_batch_end', mloss=mloss)
                # ======================= end batch ============================

            LOGGER.info(f"\ntrain_NMSE: {NMSE}\n NMSE_dB: {-1 * 10 * torch.log10(NMSE)}")

            # Scheduler
            # scheduler.step() epoch +1
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers 获取参数器中的lr
            scheduler.step()  # scheduler 下一epoch的loss

            LOGGER.debug(('%11s' * 2 + '%11.4g' * 4) %
                         (f'{epoch}/{epochs - 1}', mem, mloss[0], mloss[1], mloss[2], mloss[3]))

            # ==================== plot_save_end_epoch ===================
            if epoch % plot_epoch == 0:
                # data_iter = iter(train_loader)
                pbar = enumerate(train_loader)
                for i, (inputs_plot, labels_plot) in pbar:
                    if i != 0:
                        break
                    # inputs_plot, labels_plot = next(data_iter)
                    inputs_plot = [input.to(device) for input in inputs_plot] # [tloc, rloc]
                    # pred_plot = model(inputs_plot[0], inputs_plot[1])  # forward

                    if block:
                        block_pcd = np.loadtxt(pcd_file_path)
                        block_pcd = np.tile(np.expand_dims(block_pcd, axis=0), (min(batch_size, inputs_plot[0].shape[0]), 1, 1))

                    with torch.no_grad():
                        pred_plot = model(inputs_plot[0], inputs_plot[1], Ez_theta, s_pattern, sh_module=if_sh_module, deg=deg,
                                     block_pcd=block_pcd, rcube_=rcube_, ckai_=ckai_)

                    # labels_plot = [label.to(device) for label in labels_plot]  # [gtot, scat_gt]

                    ### model.forward(self, tloc, rloc))
                    ### Dataset_inputs: [tloc, rloc]
                    ### Dataset_labels [gtot, scat_gt]
                    ### input : [tloc, rloc]
                    ### output: [gtot, rcube_, rcube, ckai]

                    # input: Ez_pred, Ez_gt, 最后一个batch？
                    # tloc, rloc, scatter_pred, scatter_gt,ckai
                    callbacks.run('end_epoch_plot', Ez_pred=pred_plot[0].cpu().detach().numpy(), Ez_gt=labels_plot[0].cpu().detach().numpy(),
                                  tloc=inputs_plot[0].cpu().detach().numpy(), rloc=inputs_plot[1].cpu().detach().numpy(),
                                  scatter_pred=pred_plot[2].cpu().detach().numpy(), scatter_gt=labels_plot[1].cpu().detach().numpy(),
                                  ckai=pred_plot[3].cpu().detach().numpy(), epoch=epoch, save_dir=save_dir, reverse_conj=reverse_conj, train_or_val="train")  # 一个epoch


            # final_epoch 是否是最后一个epoch; noval 只val final epoch
            final_epoch = (epoch + 1 == epochs)
            if not noval or final_epoch:  # Calculate mAP
                val_loss = validate.run(batch_size=batch_size,  # 返回 acc, total loss
                                        half=amp,
                                        model=deepcopy(de_parallel(model)).eval(),
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        plots= epoch % plot_epoch==0,
                                        callbacks=callbacks,
                                        compute_loss=compute_loss,
                                        device=device,
                                        LOGGER=LOGGER,
                                        training=True,
                                        data_yml=data,
                                        pcd_file_name=pcd_file_name,
                                        epoch=epoch,
                                        reverse_conj=reverse_conj,
                                        if_Ez_theta=Ez_theta,
                                        if_s_pattern=s_pattern,
                                        scatter_BSDF=scatter_BSDF,
                                        if_sh_module=if_sh_module,
                                        deg=deg,
                                        block=block,
                                        rcube_=rcube_,
                                        ckai_=ckai_)  # forward)

            # updata best acc per epoch; 以总loss为准
            # if val_loss[0] > 1000: val_loss[0] = 10
            if val_loss[0] < mse_best or mse_best == -1.0:
                mse_best = val_loss[0]
                save_best_flag = True
                best_epoch = epoch
            else:
                save_best_flag = False


            # Save Logging results
            # 存results.csv  loss, lr
            log_vals = [val_loss, lr]  # val_loss  3*1  lr 3*1
            log_trains = mloss  # mloss 3*1
            callbacks.run('on_fit_epoch_end', log_vals, log_trains, epoch)


            # Save model
            # 存 best 模型; nosave 只存last epoch;
            if (not nosave) or (final_epoch):  # if save
                ckpt = {
                    'epoch': epoch,
                    'rcube_': rcube_,
                    'ckai_': ckai_,
                    'cur_mse': val_loss,
                    'model': deepcopy(de_parallel(model)),  # 检查模型是否是并行状态，判断并解除并行  # 存储32位float
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),  # 返回对象的 __dict__ 属性 字典格式的opt
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if save_best_flag:
                    torch.save(ckpt, best)
                # opt.save_period > 0 save every opt.save_period epochs
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt  # 节省内存 删除ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch)

            # ======================= end epoch ============================
        # ======================= end training ============================
        LOGGER.info(f'\n{epochs - start_epoch} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')


        for f in last, best:
            assert f, "final model is None!"
            if f.exists():
                # 'optimizer', 'best_fitness', 'ema', 'updates' 去除
                # 不考虑 16位 half()
                # strip_optimizer(f, LOGGER=LOGGER)  # strip optimizers  # strip optimizers 移除优化器状态 模型裁剪 去除不必要信息 存为FP16
                if f is best:  # 训练结束 validate
                    LOGGER.info(f'\nValidating {f}...')
                    loss_best = validate.run(batch_size=batch_size,
                                              half=amp,
                                              model=attempt_load(f, device, hyp['Model_type'], [centroid, m, batch_size, freq, n_scatters]),
                                              dataloader=val_loader,
                                              save_dir=save_dir,
                                              plots=not opt.noplots,
                                              callbacks=callbacks,
                                              compute_loss=compute_loss,
                                              device=device,
                                              LOGGER=LOGGER,
                                              training=False,
                                              data_yml=data,
                                              pcd_file_name=pcd_file_name,
                                              epoch=epoch,
                                              reverse_conj=reverse_conj,
                                              end_train_plot=end_train_plot,
                                              plot_scatters=plot_scatters,
                                              if_Ez_theta=Ez_theta,
                                              if_s_pattern=s_pattern,
                                              scatter_BSDF=scatter_BSDF,
                                              if_sh_module=if_sh_module,
                                              deg=deg,
                                              block=block,
                                              rcube_=rcube_,
                                              ckai_=ckai_)


        # plot loss, lr
        callbacks.run('on_train_end', best_epoch, best, epochs, loss_best, opt.noplots)


    else:
        if point_opti:
            ckpt = torch.load(test_model_path, map_location='cpu')  # load
            rcube_ = ckpt['rcube_'].to(device)  # FP32 models load to GPU
            ckai_ = ckpt['ckai_'].to(device)

        loss_best = validate.run(batch_size=batch_size,
                                 half=amp,
                                 model=attempt_load(test_model_path, device, hyp['Model_type'],
                                                    [centroid, m, batch_size, freq, n_scatters]),
                                 dataloader=val_loader,
                                 save_dir=save_dir,
                                 plots=not opt.noplots,
                                 callbacks=callbacks,
                                 compute_loss=compute_loss,
                                 device=device,
                                 LOGGER=LOGGER,
                                 training= not str(test_model_path).endswith('.pt'),
                                 data_yml=data,
                                 pcd_file_name=pcd_file_name,
                                 epoch=-1,
                                 reverse_conj=reverse_conj,
                                 end_train_plot=end_train_plot,
                                 plot_scatters=plot_scatters,
                                 if_Ez_theta=Ez_theta,
                                 if_s_pattern=s_pattern,
                                 scatter_BSDF=scatter_BSDF,
                                 if_sh_module=if_sh_module,
                                 deg=deg,
                                 block=block,
                                 rcube_=rcube_,
                                 ckai_=ckai_)


    torch.cuda.empty_cache()

    return mse_best



def main(opt, callbacks=Callbacks()):
    # available GPU
    assert torch.cuda.is_available(), f"Invalid CUDA '--device {opt.device}"
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # savedir
    # savedir = "opt.project"/"opt.name" + sep + {n} increment
    opt.project = str(opt.project)
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, sep='_', exist_ok=opt.exist_ok, mkdir=True))

    LOGGER = set_logging(opt.name, Path(opt.save_dir) / 'sensing.log', level=logging.INFO)

    copyfile(opt.hyp, os.path.join(opt.save_dir, 'opt_hyp_copy.yaml'))
    copyfile(opt.data, os.path.join(opt.save_dir, 'opt_data_copy.yaml'))

    training(opt.hyp, opt, device, callbacks, LOGGER)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default="cuda:0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str,default=r'', help='initial weights path')  
    parser.add_argument('--data', type=str, default=ROOT / 'config/data_indoor_3G_VV.yaml', help='dataset.yaml path') 
    parser.add_argument('--hyp', type=str, default=ROOT / 'config/hpy_Scatters_Green_simple.yaml', help='hyperparameters path') 

    parser.add_argument('--name', default='data_indoor_3G_VV_lr0001_bz_64', help='save to project/name') 
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs, -1 for autobatch') 

    parser.add_argument('--test_model_path', type=str, default=r'', help='if test_model_path not None, test; else training')

    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')

    parser.add_argument('--freq', type=int, default=3e9, help='freq') # 5e9 3.4e9 3e9
    parser.add_argument('--n_scatters', type=int, default=512, help='scatters number')#2048　1024
    parser.add_argument('--pcd_file_name', type=str, default="pcs_cube1_10000.txt", help='pcd path')  # pcs_cube1_5000  pcs_cube1_5000 pcs_cube1_10000  pcs_cube1
    # parser.add_argument('--pcd_file_name', type=str, default="pcs_indoor_5000.txt", help='pcd path')  # pcs_cube1_5000
    parser.add_argument('--reverse_conj', type=int, default=0, help='1: -1*conj, 2: reverse, 3: label*j  0: default; Green_complex: 1, Green_simple: 2,  new_Green_simple cube 0, indoor 1, pazhu=3')
    parser.add_argument('--Ez_theta', action='store_true', default=False, help= 'if Ez_theta, 只有simple加了, pred返回每个scatter_Ez计算loss')

    parser.add_argument('--block', action='store_true', default=True, help='True 遮挡衰减')
    parser.add_argument('--point_opti', action='store_true', default=True, help='直接优化点， 与T-R无关')

    parser.add_argument('--s_pattern', action='store_true', default=False, help='if scatter pattern, 只有simple加了，model里预测信号部分加权 gsct pattern')
    parser.add_argument('--if_sh_module', action='store_true', default=False, help='if scatter pattern, 只有simple加了，model里预测信号部分加权 gsct pattern')
    parser.add_argument('--deg', type=int, default=3, help='if scatter pattern, 只有simple加了，model里预测信号部分加权 gsct pattern')

    parser.add_argument('--KNN_mode', action='store_true', default=False, help='KNN')

    parser.add_argument('--scatter_BSDF', type=str, default='', help='if scatter pattern, scatter_BSDF, '
                                                                                              'pcd_param path 只有simple加了，model里预测信号部分加权 gsct pattern')

    parser.add_argument('--end_train_plot', action='store_true', default=True, help='train_end_plot, end train')
    parser.add_argument('--plot_scatters', action='store_true', default=False, help='plot_scatters, end train')

    parser.add_argument('--plot_epoch', type=int, default=100, help='total training epochs')
    parser.add_argument('--epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')

    # save dir increment exist-ok
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')

    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # save checkpoint every x epochs
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='optimizer') # AdamW
    parser.add_argument('--amp', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='auto mix precision, half 16 and float 32')

    # parse_known_args忽略未知的参数
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

