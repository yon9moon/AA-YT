import argparse
import collections
import torch
import random
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ==================== 完整种子锁定 ====================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =====================================================


def main(config):
    logger = config.get_logger('train')

    g = torch.Generator()
    g.manual_seed(SEED)

    # 直接使用 config 中的 data_dir（支持字符串或列表）
    train_dirs = config['data_loader']['args']['data_dir']
    logger.info(f"Loading training data from: {train_dirs}")
    data_loader = config.init_obj('data_loader', module_data, generator=g)

    # 处理验证集
    if 'valid_data_loader' not in config._config:
        # 如果 config 里没写 valid_data_loader，复制 data_loader 的配置
        valid_conf = config['data_loader'].copy()
        valid_conf['args']['shuffle'] = False
        valid_data_loader = module_data.AADataLoader(
            **valid_conf['args'],
            generator=g
        )
    else:
        valid_data_loader = config.init_obj('valid_data_loader', module_data, generator=g)

    valid_dirs = config['valid_data_loader']['args']['data_dir'] if 'valid_data_loader' in config._config else config['data_loader']['args']['data_dir']
    logger.info(f"Loading validation data from: {valid_dirs}")

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    print('device', device)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    # ================= 训练结束后清理中间 checkpoint =================
    import glob
    import os
    checkpoint_dir = config.save_dir
    # 删除所有 checkpoint-epoch*.pth 文件
    epoch_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-epoch*.pth'))
    for ckpt in epoch_checkpoints:
        try:
            os.remove(ckpt)
            logger.info(f"Removed intermediate checkpoint: {ckpt}")
        except Exception as e:
            logger.warning(f"Failed to remove {ckpt}: {e}")
    # 保留 model_best.pth（如果存在）
    best_path = os.path.join(checkpoint_dir, 'model_best.pth')
    if os.path.exists(best_path):
        logger.info(f"Kept best model: {best_path}")
    else:
        logger.info("No model_best.pth found, no best model saved.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)