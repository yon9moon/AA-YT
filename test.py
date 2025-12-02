import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from save import OutputSaver
from train import SEED


def main(config):
    logger = config.get_logger('test')

    # 在 main 函数开始时创建 OutputSaver 对象
    output_saver = OutputSaver()

    g = torch.Generator()
    g.manual_seed(SEED)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, generator=g)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    collected_outputs = []
    collected_targets = []

    with torch.no_grad():
        for i, (data, target, lens_dir) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

             # 更新预测值和真实值
            output_saver.update(output, target, {'lens_dir': lens_dir})
            collected_outputs.append(output.cpu())
            collected_targets.append(target.cpu())

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                metric_value = metric(output, target)
                if torch.is_tensor(metric_value):
                    metric_value = metric_value.detach().cpu().item()
                total_metrics[i] += metric_value * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    if collected_outputs:
        outputs = torch.cat(collected_outputs, dim=0)
        targets = torch.cat(collected_targets, dim=0)
        std_stats = module_metric.mae_with_std(outputs, targets)
        log.update({k: float(v) for k, v in std_stats.items()})

    logger.info(log)

    # 在所有预测完成后，保存数据
    output_saver.save("analysis/output.json")
    output_saver.save_classify(5, "analysis/classify_output.json")
    output_saver.save_classify_by_truth("analysis/classify_by_truth_output.json")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
