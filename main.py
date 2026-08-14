#!/usr/bin/env python
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import inspect
import random
import shutil
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from torchlight.torchlight import DictAction


def init_seed(seed):
    """
    固定随机种子，保证实验尽可能可复现。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def import_class(import_str):
    """
    根据字符串动态导入类。

    示例：
        model.kgs_gcn.Model
        feeders.feeder_penn.Feeder
    """
    mod_str, _, class_str = import_str.rpartition('.')

    if not mod_str or not class_str:
        raise ValueError(
            f'Invalid import path: {import_str}. '
            f'Expected format: package.module.Class'
        )

    __import__(mod_str)

    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError as error:
        raise ImportError(
            f'Class "{class_str}" cannot be found in module "{mod_str}".'
        ) from error


def str2bool(value):
    """
    argparse 布尔类型转换。
    """
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ('yes', 'true', 't', 'y', '1'):
        return True

    if value in ('no', 'false', 'f', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError(
        f'Unsupported boolean value: {value}'
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description='Kinematics-Driven Gaussian Splatting GCN for Penn Action'
    )

    parser.add_argument(
        '--work-dir',
        default='./work_dir/penn_action',
        help='Directory used to save logs and model checkpoints.'
    )

    parser.add_argument(
        '--model-saved-name',
        default='',
        help='Prefix used when saving model checkpoints.'
    )

    parser.add_argument(
        '--config',
        default='./config/penn_action.yaml',
        help='Path to the YAML configuration file.'
    )

    parser.add_argument(
        '--phase',
        default='train',
        choices=['train', 'test'],
        help='Run mode: train or test.'
    )

    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='Whether to save test prediction scores.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1
    )

    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='Save model every N epochs.'
    )

    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True
    )

    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='Top-k values reported at test time.'
    )

    parser.add_argument(
        '--feeder',
        default='feeders.feeder_penn.Feeder',
        help='Dataset feeder class.'
    )

    parser.add_argument(
        '--num-worker',
        type=int,
        default=8
    )

    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict()
    )

    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict()
    )

    parser.add_argument(
        '--model',
        default='model.kgs_gcn.Model',
        help='Model class path.'
    )

    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict()
    )

    parser.add_argument(
        '--weights',
        default=None,
        help='Path to pretrained model weights for test or resume.'
    )

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--step',
        type=int,
        default=[35, 55],
        nargs='+'
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='GPU id(s), e.g. --device 0 or --device 0 1.'
    )

    parser.add_argument(
        '--optimizer',
        default='SGD',
        choices=['SGD', 'Adam']
    )

    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=True
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64
    )

    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=64
    )

    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0
    )

    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005
    )

    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--warm-up-epoch',
        type=int,
        default=5
    )

    parser.add_argument(
        '--lambda-topo',
        type=float,
        default=0.1,
        help='Weight of topology consistency loss.'
    )

    return parser


class Processor:
    """
    Penn Action 训练与测试流程控制器。
    """

    def __init__(self, arg):
        self.arg = arg

        self.save_arg()

        self.output_device = (
            self.arg.device[0]
            if isinstance(self.arg.device, list)
            else self.arg.device
        )

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.output_device}')
            self.use_amp = True
        else:
            self.device = torch.device('cpu')
            self.use_amp = False

        self.print_log(f'Using device: {self.device}')

        self.global_step = 0
        self.best_acc = 0.0
        self.best_acc_epoch = 0

        self.load_model()
        self.load_optimizer()
        self.load_data()

        self.model = self.model.to(self.device)
        self.loss_ce = nn.CrossEntropyLoss().to(self.device)
        self.loss_mse = nn.MSELoss().to(self.device)

        self.scaler = GradScaler(enabled=self.use_amp)

        self.train_writer = None
        self.val_writer = None

        if self.arg.phase == 'train':
            self.arg.model_saved_name = os.path.join(
                self.arg.work_dir,
                'runs'
            )

            self.train_writer = SummaryWriter(
                os.path.join(self.arg.model_saved_name, 'train')
            )

            self.val_writer = SummaryWriter(
                os.path.join(self.arg.model_saved_name, 'val')
            )

        if (
            isinstance(self.arg.device, list)
            and len(self.arg.device) > 1
            and torch.cuda.is_available()
        ):
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=self.output_device
            )

    def save_arg(self):
        """
        保存最终参数配置到 work_dir/config.yaml。
        """
        os.makedirs(self.arg.work_dir, exist_ok=True)

        config_path = os.path.join(
            self.arg.work_dir,
            'config.yaml'
        )

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                vars(self.arg),
                f,
                default_flow_style=False,
                allow_unicode=True
            )

    def print_log(self, message, print_time=True):
        """
        控制台与日志文件同步输出。
        """
        if print_time:
            current_time = time.asctime(
                time.localtime(time.time())
            )
            message = f'[ {current_time} ] {message}'

        print(message)

        if self.arg.print_log:
            log_path = os.path.join(
                self.arg.work_dir,
                'log.txt'
            )

            with open(log_path, 'a', encoding='utf-8') as f:
                print(message, file=f)

    def load_model(self):
        """
        初始化模型并按需加载预训练权重。
        """
        Model = import_class(self.arg.model)

        model_file = inspect.getfile(Model)
        shutil.copy2(model_file, self.arg.work_dir)

        self.model = Model(**self.arg.model_args)

        if self.arg.weights is None:
            return

        if not os.path.isfile(self.arg.weights):
            raise FileNotFoundError(
                f'Weights file does not exist: {self.arg.weights}'
            )

        self.print_log(
            f'Loading weights from {self.arg.weights}.'
        )

        weights = torch.load(
            self.arg.weights,
            map_location='cpu'
        )

        cleaned_weights = OrderedDict()

        for key, value in weights.items():
            clean_key = key.replace('module.', '')
            cleaned_weights[clean_key] = value

        model_state = self.model.state_dict()

        matched_weights = {
            key: value
            for key, value in cleaned_weights.items()
            if key in model_state
            and value.shape == model_state[key].shape
        }

        missing_keys = [
            key for key in model_state.keys()
            if key not in matched_weights
        ]

        unexpected_keys = [
            key for key in cleaned_weights.keys()
            if key not in model_state
        ]

        model_state.update(matched_weights)
        self.model.load_state_dict(model_state)

        self.print_log(
            f'Loaded {len(matched_weights)} matched parameter tensors.'
        )

        if missing_keys:
            self.print_log(
                f'Warning: {len(missing_keys)} model tensors '
                f'were not loaded.'
            )

        if unexpected_keys:
            self.print_log(
                f'Warning: {len(unexpected_keys)} checkpoint tensors '
                f'are not used.'
            )

    def load_optimizer(self):
        """
        初始化优化器。
        """
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay
            )

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )

        else:
            raise ValueError(
                f'Unsupported optimizer: {self.arg.optimizer}'
            )

    def load_data(self):
        """
        构建 Penn Action 训练集和测试集。
        """
        Feeder = import_class(self.arg.feeder)

        self.data_loader = {}

        if self.arg.phase == 'train':
            train_dataset = Feeder(
                **self.arg.train_feeder_args
            )

            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=init_seed
            )

        test_dataset = Feeder(
            **self.arg.test_feeder_args
        )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=init_seed
        )

    def adjust_learning_rate(self, epoch):
        """
        Warm-up + step decay 学习率策略。
        """
        if epoch < self.arg.warm_up_epoch:
            lr = (
                self.arg.base_lr
                * (epoch + 1)
                / self.arg.warm_up_epoch
            )
        else:
            lr = self.arg.base_lr * (
                self.arg.lr_decay_rate
                ** np.sum(epoch >= np.asarray(self.arg.step))
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def calculate_topology_loss(self, learned_topos, A_prior):

        topology_loss = 0.0

        for topo in learned_topos:
            learned_adjacency = torch.sigmoid(
                topo.mean(dim=1)
            )

            topology_loss = topology_loss + self.loss_mse(
                learned_adjacency,
                A_prior.detach()
            )

        return topology_loss

    def train(self, epoch, save_model=False):
        """
        执行一个 epoch 的训练。
        """
        self.model.train()

        self.print_log(
            f'Training epoch: {epoch + 1}'
        )

        lr = self.adjust_learning_rate(epoch)

        if self.train_writer is not None:
            self.train_writer.add_scalar(
                'Train/LR',
                lr,
                epoch + 1
            )

        loss_values = []
        ce_loss_values = []
        topo_loss_values = []
        acc_values = []

        loader = self.data_loader['train']
        process = tqdm(loader, ncols=100)

        for batch_idx, (data, label, _) in enumerate(process):
            self.global_step += 1

            data = data.float().to(
                self.device,
                non_blocking=True
            )

            label = label.long().to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output, learned_topos, A_prior = self.model(data)

                loss_ce = self.loss_ce(output, label)

                loss_topo = self.calculate_topology_loss(
                    learned_topos,
                    A_prior
                )

                # 前 5 个 epoch 对拓扑损失进行逐步 warm-up。
                topology_weight = self.arg.lambda_topo * min(
                    1.0,
                    epoch / 5.0
                )

                loss = loss_ce + topology_weight * loss_topo

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            prediction = torch.argmax(output, dim=1)
            accuracy = torch.mean(
                (prediction == label).float()
            )

            loss_values.append(loss.item())
            ce_loss_values.append(loss_ce.item())
            topo_loss_values.append(loss_topo.item())
            acc_values.append(accuracy.item())

            if self.train_writer is not None:
                self.train_writer.add_scalar(
                    'Train/Loss',
                    loss.item(),
                    self.global_step
                )

                self.train_writer.add_scalar(
                    'Train/CrossEntropyLoss',
                    loss_ce.item(),
                    self.global_step
                )

                self.train_writer.add_scalar(
                    'Train/TopologyLoss',
                    loss_topo.item(),
                    self.global_step
                )

                self.train_writer.add_scalar(
                    'Train/Accuracy',
                    accuracy.item(),
                    self.global_step
                )

            process.set_postfix(
                loss=f'{loss.item():.4f}',
                acc=f'{accuracy.item() * 100:.2f}%'
            )

        mean_loss = float(np.mean(loss_values))
        mean_ce_loss = float(np.mean(ce_loss_values))
        mean_topo_loss = float(np.mean(topo_loss_values))
        mean_acc = float(np.mean(acc_values))

        self.print_log(
            f'\tMean training loss: {mean_loss:.4f}. '
            f'CE loss: {mean_ce_loss:.4f}. '
            f'Topology loss: {mean_topo_loss:.4f}. '
            f'Train acc: {mean_acc * 100:.2f}%.'
        )

        if save_model:
            self.save_model(epoch)

    def save_model(self, epoch):
        """
        保存模型参数。
        """
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        weights = OrderedDict(
            (key, value.detach().cpu())
            for key, value in state_dict.items()
        )

        save_path = (
            f'{self.arg.model_saved_name}'
            f'-{epoch + 1}.pt'
        )

        torch.save(weights, save_path)

        self.print_log(
            f'Model saved to: {save_path}'
        )

    def eval(self, epoch=0, save_score=False, loader_name='test'):
        """
        在测试集上评估模型。
        """
        self.model.eval()

        self.print_log(
            f'Evaluating on {loader_name} set...'
        )

        loss_values = []
        score_fragments = []
        label_fragments = []

        loader = self.data_loader[loader_name]
        process = tqdm(loader, ncols=100)

        with torch.no_grad():
            for data, label, index in process:
                data = data.float().to(
                    self.device,
                    non_blocking=True
                )

                label = label.long().to(
                    self.device,
                    non_blocking=True
                )

                with autocast(enabled=self.use_amp):
                    output, _, _ = self.model(data)
                    loss = self.loss_ce(output, label)

                loss_values.append(loss.item())

                score_fragments.append(
                    output.detach().cpu().numpy()
                )

                label_fragments.append(
                    label.detach().cpu().numpy()
                )

        score = np.concatenate(score_fragments, axis=0)
        labels = np.concatenate(label_fragments, axis=0)

        mean_loss = float(np.mean(loss_values))

        self.print_log(
            f'\tMean test loss: {mean_loss:.4f}'
        )

        topk_results = {}

        for k in self.arg.show_topk:
            accuracy = self.data_loader[
                loader_name
            ].dataset.top_k(score, k)

            topk_results[k] = accuracy

            self.print_log(
                f'\tTop{k}: {accuracy * 100:.2f}%'
            )

        top1_acc = topk_results.get(
            1,
            list(topk_results.values())[0]
        )

        if self.val_writer is not None:
            self.val_writer.add_scalar(
                'Test/Loss',
                mean_loss,
                epoch + 1
            )

            self.val_writer.add_scalar(
                'Test/Top1Accuracy',
                top1_acc,
                epoch + 1
            )

        if top1_acc > self.best_acc:
            self.best_acc = top1_acc
            self.best_acc_epoch = epoch + 1

            self.print_log(
                f'Current best Top-1 accuracy: '
                f'{self.best_acc * 100:.2f}% '
                f'at epoch {self.best_acc_epoch}.'
            )

        if save_score:
            score_dict = {
                index: score[index]
                for index in range(len(score))
            }

            score_path = os.path.join(
                self.arg.work_dir,
                f'epoch{epoch + 1}_{loader_name}_score.pkl'
            )

            import pickle

            with open(score_path, 'wb') as f:
                pickle.dump(score_dict, f)

            self.print_log(
                f'Test scores saved to: {score_path}'
            )

        return top1_acc, mean_loss, labels, score

    def start(self):
        """
        根据 phase 启动训练或测试。
        """
        if self.arg.phase == 'train':
            self.print_log(
                f'Parameters:\n{vars(self.arg)}\n',
                print_time=False
            )

            for epoch in range(
                self.arg.start_epoch,
                self.arg.num_epoch
            ):
                save_model = (
                    (epoch + 1) % self.arg.save_interval == 0
                    or (epoch + 1) == self.arg.num_epoch
                )

                self.train(
                    epoch=epoch,
                    save_model=save_model
                )

                self.eval(
                    epoch=epoch,
                    save_score=self.arg.save_score,
                    loader_name='test'
                )

            self.print_log(
                f'Training finished. '
                f'Best Top-1 accuracy: {self.best_acc * 100:.2f}% '
                f'at epoch {self.best_acc_epoch}.'
            )

            if self.train_writer is not None:
                self.train_writer.close()

            if self.val_writer is not None:
                self.val_writer.close()

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError(
                    'Testing requires --weights or weights in YAML config.'
                )

            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name='test'
            )


if __name__ == '__main__':
    parser = get_parser()

    initial_args = parser.parse_args()

    if initial_args.config is not None:
        if not os.path.isfile(initial_args.config):
            raise FileNotFoundError(
                f'Config file does not exist: {initial_args.config}'
            )

        with open(
            initial_args.config,
            'r',
            encoding='utf-8'
        ) as f:
            default_arg = yaml.safe_load(f)

        if default_arg is None:
            default_arg = {}

        parser_keys = vars(initial_args).keys()

        for key in default_arg.keys():
            if key not in parser_keys:
                raise KeyError(
                    f'Unknown argument "{key}" in config file '
                    f'{initial_args.config}.'
                )

        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    init_seed(arg.seed)

    processor = Processor(arg)
    processor.start()
