import json
import os
import yaml
import datetime
import logging
import time
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SimpleLinear
from config import get_config
from logger import setup_logging
from utils import AverageMeter
from visualization import plot_heat, plot_loss, plot_info

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cfg', type=str, default='configs/f_principle.yml',
                        help='which config to use')
    parser.add_argument('--tag', type=str, default=None)
    # TODO
    return parser.parse_args() 

        
def train_one_epoch(config, model, target_fn, dataloader, optimizer, epoch, lr_scheduler):
    f"""
    params:
        config: parameters
        model: DNN
        targte_fn: function to fit
        dataloader: train data only contains x
        optimizer:
        epoch:
        lr_schedule:
    """
    device = torch.device('cuda', 0)
    model.to(device)
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(dataloader)
    
    batch_meter = AverageMeter()
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    grad_meter = AverageMeter()
    
    start_time = time.time()
    
    for step, x in enumerate(dataloader):
        # print(step)
        x = x.to(device)
        x.requires_grad_(True)
        y = target_fn(x)
        
        y_pred = model(x)
        
        # TODO
        dy_dx = torch.autograd.grad(torch.sum(y), x, create_graph=True, retain_graph=True)
        dy_dx_pred = torch.autograd.grad(torch.sum(y_pred), x, create_graph=True, retain_graph=True)
        
        recon_loss = F.mse_loss(y_pred, y)
        grad_loss = F.mse_loss(dy_dx_pred[0], dy_dx[0])
        loss = recon_loss + grad_loss
        
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), x.shape[0])
        recon_meter.update(recon_loss.item(), x.shape[0])
        grad_meter.update(grad_loss.item(), x.shape[0])
        batch_meter.update(time.time() - start_time)
        
        if step % config.RESULTS.PRINT_FREQ == 0:
            etas = batch_meter.avg * (num_steps - step)
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logging.info(
                f"Train: {epoch}/{config.TRAIN.EPOCHS}, {step}/{num_steps} "
                f"eta: {datetime.timedelta(seconds=int(etas))} "
                f"time: {batch_meter.val:.5f}, {batch_meter.avg:.5f} "
                f"loss: {loss_meter.val:.5f}, {loss_meter.avg:.5f} "
                f"recon_loss: {recon_meter.val:.5f}, {recon_meter.avg:.5f} "
                f"grad_loss: {grad_meter.val:.5f}, {grad_meter.avg:.5f} "
                f"mem {memory_used:.4f}MB"
            )
    epoch_time = time.time() - start_time
    logging.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    lr_scheduler.step()
    
    

@torch.no_grad()
def validate(model, dataloader, target_fn):
    device = torch.device('cuda', 0)
    model.eval()
    
    x = []
    y = []
    y_pred = []
    for input in dataloader:
        input = input.to(device)
        target = target_fn(input)
        
        output = model(input)
        
        x.append(input.detach().cpu().numpy())
        y.append(target.detach().cpu().numpy())
        y_pred.append(output.detach().cpu().numpy())
    
    x = np.concatenate(x, axis=0)[:, 0]
    index = np.argsort(x)
    x = x[index]
    y = np.concatenate(y, axis=0)[:, 0]
    y = y[index]
    y_pred = np.concatenate(y_pred, axis=0)[:, 0]
    y_pred = y_pred[index]
    
    return x, y, y_pred
        
def main(config):    
    def target_fn(x):
        return torch.sin(x) + torch.sin(3*x) + torch.sin(5*x)
    
    model = SimpleLinear(
        config.MODEL.IN_FEAT,
        config.MODEL.OUT_FEAT,
        config.MODEL.LAYERS,
        config.MODEL.ACT
    ).to(torch.device('cuda', 0))
    train_x = np.linspace(
        config.DATA.START, config.DATA.END, config.DATA.TRAIN_SIZE, endpoint=True
    ).reshape(-1, 1).astype(np.float32)
    
    test_x = np.linspace(
        config.DATA.START, config.DATA.END, config.DATA.TEST_SIZE, endpoint=True
    ).reshape(-1, 1).astype(np.float32)
    
    train_dataloader = DataLoader(train_x, batch_size=config.DATA.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_x, batch_size=config.DATA.BATCH_SIZE, shuffle=True)
    # TODO: self-define optimizer
    optimizer = torch.optim.__dict__[config.TRAIN.OPT](filter(lambda p: p.requires_grad, list(model.parameters())), lr=config.TRAIN.LR)
    # TODO: self-define lr_schedule
    lr_schduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    
    y_pred_total = []
    for epoch in range(config.TRAIN.EPOCHS):
        train_one_epoch(config, model, target_fn, train_dataloader, optimizer, epoch, lr_schduler)
        
        train_x, train_y, train_y_pred = validate(model, train_dataloader, target_fn)
        
        y_pred_total.append(train_y_pred)
        if epoch % config.RESULTS.EVAL_FREQ == 0:
            plot_info(epoch, train_x, train_y, train_y_pred, save_path=config.RESULTS.LOGS_DIR, mode='train')
            test_x, test_y, test_y_pred = validate(model, test_dataloader, target_fn)
            plot_info(epoch, test_x, test_y, test_y_pred, save_path=config.RESULTS.LOGS_DIR, mode='test')
        
        if epoch != 0 and epoch % config.RESULTS.PLOT_HEAT == 0:
            plot_heat(train_x, train_y, np.squeeze(y_pred_total), save_path=config.RESULTS.LOGS_DIR)
    
    

if __name__ == "__main__":
    args = get_parser()
    config = get_config(args)
    setup_logging(args, config)
    
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        json.dump(config, f)
    logging.info(f"Full config saved to {path}")
    
    main(config)
