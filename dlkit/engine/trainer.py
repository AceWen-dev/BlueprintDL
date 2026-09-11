import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import yaml

from tqdm import tqdm

import dlkit.data  # noqa: F401
import dlkit.models  # noqa: F401
import dlkit.metrics  # noqa: F401
from dlkit.registry import build_from_cfg, OPTIMIZERS, SCHEDULERS
from dlkit.data.builder import build_dataloaders
from dlkit.utils.seed import set_seed
from dlkit.utils.device import resolve_device
from dlkit.utils.logger import get_logger
from dlkit.utils.checkpoint import load_checkpoint
from dlkit.utils.batch import to_device


class Trainer:
    def __init__(self, cfg, work_dir=None, device=None, resume_from=None, logger=None):
        self.cfg = cfg
        self.work_dir = work_dir or cfg.get('output_dir', 'runs/default')
        os.makedirs(self.work_dir, exist_ok=True)
        self.device = resolve_device(device or cfg.get('device', 'auto'))
        self.resume_from = resume_from
        self.logger = logger or get_logger(self.work_dir)

        self.hooks = defaultdict(list)#创建一个值为空列表的字典，defaultdict键不存在时自动调用list()创建一个空列表补充

        self.stop_training = False
        self.best_metric = None
        self.metric_history = []
        self.current_epoch = 0
        self.start_epoch = 0
        self._built = False

    def register_hook(self, event, fn): #把fn这个函数添加到self.hooks字典中，键是event，值是一个列表，里面存放了所有注册的函数
        self.hooks[event].append(fn)

    def _fire(self, event, **kwargs): #把event对应的函数列表取出来，依次调用这些函数，传入self和kwargs
        for fn in self.hooks.get(event, []):
            fn(self, **kwargs)

    def _build(self):
        set_seed(self.cfg.get('seed', 42))    #从配置文件中取seed的值如果没有直接默认42
        self.logger.info('Device: %s', self.device)  #

        self.train_loader, self.val_loader = build_dataloaders(self.cfg['data']) #数据预处理，搭建训练集和验证集的dataloader，返回的是两个dataloader对象
        self.logger.info(
            'Data: %d train samples, %d val samples',
            len(self.train_loader.dataset), len(self.val_loader.dataset),
        )

        self.model = build_from_cfg(self.cfg['model']).to(self.device) #搭建模型架构    
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info('Model %s: %.2fM params', type(self.model).__name__, n_params / 1e6) #搭建训练日至

        self.criterion = build_from_cfg(self.cfg['loss']).to(self.device) #定义损失函数
        self.optimizer = self._build_optimizer() #搭建优化器 单独写方法
        self.scheduler = self._build_scheduler() #搭建学习率调度器 单独写方法
        self.metrics = [build_from_cfg(m) for m in self.cfg.get('metrics', [])] #定义评价指标

        train_cfg = self.cfg['train'] 
        sb = train_cfg.get('save_best')
        self.save_best_name = sb.get('name') if sb else None
        self.save_best_mode = sb.get('mode', 'max') if sb else 'max'

        with open(os.path.join(self.work_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)

        if self.resume_from:
            self._resume(self.resume_from)

        self._built = True

    def _build_optimizer(self):
        opt_cfg = dict(self.cfg['optimizer'])
        opt_type = opt_cfg.pop('type')
        opt_params = build_from_cfg(opt_cfg.get('params', {}))
        return OPTIMIZERS.get(opt_type)(self.model.parameters(), **opt_params)

    def _build_scheduler(self):
        s_cfg = self.cfg.get('scheduler')
        if not s_cfg:
            return None
        s_type = s_cfg['type']
        params = dict(s_cfg.get('params', {}) or {})
        cls = SCHEDULERS.get(s_type)
        if s_type == 'PolyLR':
            total_steps = len(self.train_loader) * int(self.cfg['train']['epochs'])
            params.setdefault('max_iters', total_steps)
        return cls(self.optimizer, **params)

    def _resume(self, path):
        self.logger.info('Resuming from %s', path)
        ckpt = load_checkpoint(
            path, model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, device=self.device,
        )
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_metric = ckpt.get('best_metric')
        self.metric_history = ckpt.get('metric_history', [])

    def train(self):
        if not self._built:
            self._build()
        train_cfg = self.cfg['train']  
        epochs = int(train_cfg['epochs'])

        self._fire('before_train')
        self.logger.info('Training %d epochs (start=%d)', epochs, self.start_epoch)
        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch
            self._fire('before_epoch', epoch=epoch)
            train_loss = self._train_one_epoch(epoch, train_cfg)
            val_metrics = self._validate()
            val_metrics['train_loss'] = train_loss
            self.metric_history.append({'epoch': epoch, **val_metrics})
            self._write_history()

            self.logger.info(
                'Epoch %d | train_loss=%.4f | %s',
                epoch, train_loss,
                ' '.join('%s=%.4f' % (k, v) for k, v in val_metrics.items()
                         if isinstance(v, float) and k != 'train_loss'),
            )

            if self.scheduler is not None and not getattr(self.scheduler, 'per_iter', False):
                self.scheduler.step()

            self._save_checkpoints(epoch)
            self._fire('after_epoch', epoch=epoch, metrics=val_metrics)

            if self.stop_training:
                self.logger.info('Early stopping triggered at epoch %d', epoch)
                break

        self._fire('after_train')
        return self.metric_history

    def _train_one_epoch(self, epoch, train_cfg):
        self.model.train() #把模型设置为训练模式，启用dropout和batchnorm
        use_amp = bool(train_cfg.get('amp', False)) and self.device.type == 'cuda' #设置混合精度训练，只有在cuda设备上才启用
        grad_clip = train_cfg.get('gradient_clip') # 梯度裁剪，防止梯度爆炸
        log_interval = int(train_cfg.get('log_interval', 10)) # 人工设置的日志打印间隔，默认10个batch打印一次
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None #  梯度缩放器，防止梯度下溢，只有在启用混合精度训练时才使用

        running = 0.0
        pbar = tqdm(self.train_loader, desc='Epoch %d' % epoch) #给循环加进度条显示，desc是进度条前的描述信息 %d是占位符，%就是格式化输出，后面跟着的epoch会替换掉%d
        for step, batch in enumerate(pbar): #dtaloader返回的是一个batch的字典，里面包含了image和label等信息,不同任何返回的内容也不同大致就是原图和标签
            self._fire('before_train_step', batch=batch, step=step)
            batch = to_device(batch, self.device)
            images = batch['image'] #取出batch字典中的image键对应的值，作为输入

            self.optimizer.zero_grad() #清除梯度缓存，避免梯度累加
            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    preds = self.model(images)
                    loss = self.criterion(preds, batch)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                preds = self.model(images) # 前向传播，得到预测结果
                loss = self.criterion(preds, batch)# 计算损失函数，preds是模型的输出，batch是包含真实标签的字典
                loss.backward() # 反向传播，计算梯度
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step() # 更新模型参数，使用优化器根据计算得到的梯度来更新模型的权重

            if self.scheduler is not None and getattr(self.scheduler, 'per_iter', False): # 如果学习率调度器存在并且设置为每个迭代更新，则调用scheduler.step()来更新学习率
                self.scheduler.step()

            loss_value = float(loss.detach().item()) # detach()方法将loss从计算图中分离出来，item()方法将其转换为Python的标量值，float()确保它是一个浮点数
            running += loss_value # 累加每个batch的损失值，用于计算平均损失
            if step % log_interval == 0: # 每隔log_interval个batch打印一次日志信息
                pbar.set_postfix(loss='%.4f' % (running / (step + 1)))
            self._fire('after_train_step', loss=loss_value, step=step)   # 

        return running / max(1, len(self.train_loader)) # 返回平均损失

    def _validate(self):
        self.model.eval()
        for m in self.metrics:
            m.reset()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = to_device(batch, self.device)
                images = batch['image']
                preds = self.model(images)
                loss = self.criterion(preds, batch)
                total_loss += float(loss.detach().item())
                n += 1
                for m in self.metrics:
                    m.update(preds, batch)
        result = {'val_loss': float(total_loss / max(1, n))}
        for m in self.metrics:
            for k, v in m.compute().items():
                if isinstance(v, float):
                    result[k] = float(v)
        return result

    def _write_history(self):
        with open(os.path.join(self.work_dir, 'metrics.jsonl'), 'w', encoding='utf-8') as f:
            for entry in self.metric_history:
                f.write(json.dumps(entry) + '\n')

    def _save_checkpoints(self, epoch):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_metric': self.best_metric,
            'metric_history': self.metric_history,
            'cfg': self.cfg,
        }
        torch.save(ckpt, os.path.join(self.work_dir, 'last.pth'))

        if self.save_best_name is None:
            return
        current = self.metric_history[-1].get(self.save_best_name)
        if current is None:
            return
        if self.best_metric is None:
            better = True
        elif self.save_best_mode == 'max':
            better = current > self.best_metric
        else:
            better = current < self.best_metric
        if better:
            self.best_metric = current
            torch.save(ckpt, os.path.join(self.work_dir, 'best.pth'))
            self.logger.info('Saved best checkpoint (epoch=%d, %s=%.4f)', epoch, self.save_best_name, current)
