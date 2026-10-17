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
import dlkit.engine.recipes  # noqa: F401  注册内置 Recipe(EMA 教师等)
from dlkit.builders import (
    build_dataloaders,
    build_from_cfg,
    build_optimizer,
    build_scheduler,
)
from dlkit.utils.seed import set_seed
from dlkit.utils.device import resolve_device
from dlkit.utils.logger import get_logger
from dlkit.utils.checkpoint import load_checkpoint
from dlkit.utils.batch import to_device


class Trainer:
    def __init__(self, cfg, work_dir=None, device=None, resume_from=None, logger=None, opts=None):
        self.cfg = cfg
        self.opts = list(opts or [])  # 命令行覆盖项,写入 run.json 供复现
        self.device = resolve_device(device or cfg.get('device', 'auto'))
        self.resume_from = resume_from
        self.init_from = (cfg.get('train') or {}).get('init_from')  # 仅加载权重(微调)
        self.loaded_plugins = []
        self.recipe = None
        self._resume_missed = False   # resume=latest 但找不到可续接的 run
        self._resume_continued = False  # resume=latest 已续接到既有谱系
        self.work_dir = self._resolve_work_dir(work_dir)  # 解析运行产物目录
        os.makedirs(self.work_dir, exist_ok=True) #创建工作目录
        self._owns_logger = logger is None
        self.logger = logger or get_logger(self.work_dir)
        self.run_manifest = self._init_manifest()

        self.hooks = defaultdict(list)#创建一个值为空列表的字典，defaultdict键不存在时自动调用list()创建一个空列表补充

        self.stop_training = False
        self.best_metric = None
        self.best_epoch = None
        self.metric_history = []
        self.current_epoch = 0
        self.start_epoch = 0
        self._built = False

    def _resolve_work_dir(self, override):
        """解析本次训练的产物目录。

        规则:
        - 显式 work_dir(--work-dir)始终优先;
        - resume=latest 时,自动续接同一 <project>/<experiment> 下最近的
          时间戳目录(谱系延续,metrics 历史保持同一条血统);
        - 其余情况创建 runs/<project>/<experiment>/<时间戳>/ 唯一目录,
          同一实验重跑永不覆盖。
        """

        if override:
            return override

        from dlkit.utils import runs as _runs

        project = str(self.cfg.get('project') or 'default')
        experiment = str(self.cfg.get('name') or self.cfg.get('experiment') or 'unnamed')
        runs_root = self.cfg.get('runs_root', _runs.RUNS_ROOT_DEFAULT)

        if self.resume_from == 'latest':
            last = _runs.latest_run(runs_root, project, experiment)
            if last is not None:
                self.resume_from = str(last / 'last.pth')
                self._resume_continued = True
                return last
            self._resume_missed = True
            return _runs.make_run_dir(runs_root, project, experiment)

        return _runs.make_run_dir(runs_root, project, experiment)

    def _init_manifest(self):
        """建立/更新 run.json:记录配置快照、命令覆盖与环境信息。"""

        from dlkit.utils import runs as _runs
        from dlkit.utils.runs import (
            init_run_manifest,
            read_run_manifest,
            write_run_manifest,
        )

        environment = _runs.collect_environment_info(
            device=str(self.device),
            torch_version=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
        )
        overrides = {}
        for item in self.opts:
            key, _, value = item.partition('=')
            overrides[key.strip()] = value
        resume_from = self.resume_from if self.resume_from and self.resume_from != 'latest' else None

        existing = read_run_manifest(self.work_dir)
        if existing is not None:
            # 续接已有谱系:保留原清单,刷新状态与续训信息
            existing['status'] = 'running'
            existing['resume_from'] = resume_from
            existing['init_from'] = self.init_from
            existing.setdefault('raw_opts', [])
            existing['overrides'] = {**existing.get('overrides', {}), **overrides}
            existing['environment'] = {**existing.get('environment', {}), **environment}
            write_run_manifest(self.work_dir, existing)
            return existing

        return init_run_manifest(
            self.work_dir,
            project=str(self.cfg.get('project') or 'default'),
            experiment=str(self.cfg.get('name') or self.cfg.get('experiment') or 'unnamed'),
            config=self.cfg,
            overrides=overrides,
            raw_opts=self.opts,
            resume_from=resume_from,
            init_from=self.init_from,
            environment=environment,
        )

    def register_hook(self, event, fn): #把fn这个函数添加到self.hooks字典中，键是event，值是一个列表，里面存放了所有注册的函数
        self.hooks[event].append(fn)

    def _fire(self, event, **kwargs): #把event对应的函数列表取出来，依次调用这些函数，传入self和kwargs
        for fn in self.hooks.get(event, []):
            fn(self, **kwargs)

    def _build(self):
        self._load_plugins()                   # 先加载项目插件(自定义组件/Recipe)
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
        self.optimizer = build_optimizer(
            self.cfg['optimizer'], parameters=self.model.parameters()
        )
        total_steps = len(self.train_loader) * int(self.cfg['train']['epochs'])
        self.scheduler = build_scheduler(
            self.cfg.get('scheduler'),
            optimizer=self.optimizer,
            total_steps=total_steps,
        )
        self.metrics = [build_from_cfg(m) for m in self.cfg.get('metrics', [])] #定义评价指标

        train_cfg = self.cfg['train'] 
        sb = train_cfg.get('save_best')
        self.save_best_name = sb.get('name') if sb else None
        self.save_best_mode = sb.get('mode', 'max') if sb else 'max'

        with open(os.path.join(self.work_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)

        if self.resume_from:
            self._resume(self.resume_from)
        elif self.init_from:
            self._apply_init_from()

        recipe_cfg = self.cfg.get('recipe')
        if recipe_cfg:
            self.recipe = build_from_cfg(recipe_cfg)
            if not hasattr(self.recipe, 'attach'):
                raise TypeError(
                    'recipe %r must implement attach(trainer)'
                    % type(self.recipe).__name__
                )
            self.recipe.attach(self)
            self.logger.info('Recipe attached: %s', type(self.recipe).__name__)

        self._built = True

    def _load_plugins(self):
        """导入 config['imports'] 声明的项目插件,并把来源写进 run.json。"""

        entries = self.cfg.get('imports') or []
        if not entries:
            return
        from dlkit.utils.plugins import load_plugin_modules
        from dlkit.utils.runs import write_run_manifest

        self.loaded_plugins = load_plugin_modules(entries)
        if not self.loaded_plugins:
            return
        self.run_manifest['plugins'] = self.loaded_plugins
        write_run_manifest(self.work_dir, self.run_manifest)
        self.logger.info('Loaded plugin modules: %s', ', '.join(self.loaded_plugins))

    def _apply_init_from(self):
        """仅加载模型权重(迁移学习/微调),不恢复 epoch 与 optimizer 状态。"""

        from dlkit.utils.checkpoint import load_model_weights

        self.logger.info('Initializing weights from %s (weights-only, no resume)', self.init_from)
        result = load_model_weights(self.init_from, self.model, device=self.device)
        missing = list(getattr(result, 'missing_keys', []) or [])
        unexpected = list(getattr(result, 'unexpected_keys', []) or [])
        matched = len(self.model.state_dict()) - len(missing)
        self.logger.info(
            'init_from: matched=%d missing=%d unexpected=%d',
            matched, len(missing), len(unexpected),
        )
        if missing:
            self.logger.warning('init_from missing keys (kept random init): %s%s',
                                ', '.join(missing[:8]), ' ...' if len(missing) > 8 else '')
        if unexpected:
            self.logger.warning('init_from unexpected keys (ignored): %s%s',
                                ', '.join(unexpected[:8]), ' ...' if len(unexpected) > 8 else '')

    def _resume(self, path): # 恢复训练，加载模型权重，优化器状态，调度器状态，当前epoch，最佳评价指标，历史评价指标等信息
        self.logger.info('Resuming from %s', path)
        ckpt = load_checkpoint(
            path, model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, device=self.device,
        )
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_metric = ckpt.get('best_metric')
        self.best_epoch = ckpt.get('best_epoch')
        self.metric_history = ckpt.get('metric_history', [])

    def train(self):
        """训练入口:结束后自动关闭本 run 的日志文件句柄。"""

        try:
            return self._train_impl()
        finally:
            self.close()

    def close(self):
        """关闭日志文件句柄(Windows 下释放 log.txt,允许移动/删除 run 目录)。"""

        if getattr(self, '_owns_logger', False):
            from dlkit.utils.logger import close_logger

            close_logger(self.logger)
            self._owns_logger = False

    def _train_impl(self):
        if not self._built:
            self._build()
        train_cfg = self.cfg['train']  
        epochs = int(train_cfg['epochs'])

        if self._resume_missed:
            self.logger.warning('resume=latest: no previous run found for %s/%s; starting fresh',
                                self.cfg.get('project'), self.cfg.get('name'))

        self._fire('before_train')
        self.logger.info('Training %d epochs (start=%d)', epochs, self.start_epoch)
        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch
            self._fire('before_epoch', epoch=epoch)
            train_loss = self._train_one_epoch(epoch, train_cfg)
            self._fire('before_validation', epoch=epoch)
            val_metrics = self._validate()
            self._fire('after_validation', epoch=epoch, metrics=val_metrics)
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
        self._finalize_manifest()
        return self.metric_history

    def _finalize_manifest(self):
        """把最终与 best 指标回填到 run.json。"""

        from dlkit.utils.runs import finalize_run_manifest

        final = dict(self.metric_history[-1]) if self.metric_history else {}
        total_epochs = int(final.get('epoch', -1)) + 1
        best = None
        if self.save_best_name is not None and self.best_metric is not None:
            best = {
                'name': self.save_best_name,
                'value': float(self.best_metric),
                'epoch': self.best_epoch,
            }
        self.run_manifest = finalize_run_manifest(
            self.work_dir,
            total_epochs=total_epochs,
            last_metrics=final,
            save_best=best,
        )

    def _forward_model(self, batch):
        """模型前向协议。

        - 若模型实现 ``forward_batch(batch)``(多模态等需要额外输入的场景),
          由模型自己从 batch 取数据;
        - 否则保持既有契约:``model(batch['image'])``。
        """

        forward_batch = getattr(self.model, 'forward_batch', None)
        if callable(forward_batch):
            return forward_batch(batch)
        return self.model(batch['image'])

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

            self.optimizer.zero_grad() #清除梯度缓存，避免梯度累加
            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    preds = self._forward_model(batch)
                    loss = self.criterion(preds, batch)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                preds = self._forward_model(batch) # 前向传播，得到预测结果
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

        return running / max(1, len(self.train_loader)) # 返回平均损失，模型是一个批次进去计算损失，然后更新一次参数，len(self.train_loader)训练集有多少个batch，max(1 防止分母为零

    def _validate(self): #
        self.model.eval() #把模型切换为验证模式
        for m in self.metrics:
            m.reset() # 把所有评价指标清零
        total_loss = 0.0
        n = 0
        with torch.no_grad(): #不用计算梯度进行推理验证
            for batch in self.val_loader:
                batch = to_device(batch, self.device)
                preds = self._forward_model(batch)
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

    def _write_history(self): #记录训练的历史记录
        with open(os.path.join(self.work_dir, 'metrics.jsonl'), 'w', encoding='utf-8') as f:
            for entry in self.metric_history:
                f.write(json.dumps(entry) + '\n')

    def _save_checkpoints(self, epoch): #保存模型权重 ，配置文件，评价指标，其中优化器，调度器是模型恢复训练时需要用到的
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
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
            self.best_epoch = epoch
            torch.save(ckpt, os.path.join(self.work_dir, 'best.pth'))
            self.logger.info('Saved best checkpoint (epoch=%d, %s=%.4f)', epoch, self.save_best_name, current)
