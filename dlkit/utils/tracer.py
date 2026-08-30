"""张量数据流可视化工具。

在任意 nn.Module 上挂 forward hook，按前向执行顺序打印每一层的
输入/输出张量形状，帮助理解模型中张量是如何流动和变化的。

用法一（一行搞定）:
    from dlkit.utils.tracer import trace
    trace(model, input_shape=(1, 3, 256, 256))

用法二（挂上 hook 后自己前向，可同时拿到模型输出）:
    from dlkit.utils.tracer import TensorTracer
    tracer = TensorTracer(model, include='conv')
    out = model(x)
    tracer.print_report()
    tracer.remove()

用法三（直接从配置文件构建并追踪）:
    from dlkit.utils.tracer import trace_from_config
    trace_from_config('configs/unet_toy.yaml', input_shape=(1, 3, 256, 256))
"""

import torch
import torch.nn as nn


_PRESETS = {
    'conv': (nn.Conv1d, nn.Conv2d, nn.Conv3d),
    'bn': (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
    'pool': (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d),
    'activation': (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.Sigmoid, nn.Tanh),
    'linear': (nn.Linear,),
}


def _resolve_types(spec):
    if spec is None or spec == 'all':
        return None
    if isinstance(spec, str):
        if spec not in _PRESETS:
            raise ValueError('未知过滤名 %r，可选: %s' % (spec, sorted(_PRESETS)))
        return _PRESETS[spec]
    if isinstance(spec, type):
        return (spec,)
    if isinstance(spec, (tuple, list)):
        return tuple(spec)
    raise TypeError('不支持的 include/exclude 参数: %r' % (spec,))


def _shape_of(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_of(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_of(v) for k, v in x.items()}
    return type(x).__name__


def _fmt_inputs(inputs):
    shapes = [_shape_of(i) for i in inputs]
    if len(shapes) == 1:
        return str(shapes[0])
    return str(shapes)


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


class TensorTracer:
    """在模型各层挂 forward hook，记录并打印张量形状。"""

    def __init__(self, model, include=None, exclude=None, max_depth=None):
        self.model = model
        self.include_types = _resolve_types(include)
        self.exclude_types = _resolve_types(exclude)
        self.max_depth = max_depth
        self.records = []
        self._handles = []
        self._register()

    def _should_trace(self, name, module):
        if self.include_types is not None and not isinstance(module, self.include_types):
            return False
        if self.exclude_types is not None and isinstance(module, self.exclude_types):
            return False
        if self.max_depth is not None:
            depth = name.count('.') + 1 if name else 0
            if depth > self.max_depth:
                return False
        return True

    def _register(self):
        for name, module in self.model.named_modules():
            if self._should_trace(name, module):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.records.append(
                {
                    'name': name,
                    'type': module.__class__.__name__,
                    'in': _fmt_inputs(inputs),
                    'out': _shape_of(output),
                    'params': _n_params(module),
                }
            )

        return hook

    def clear(self):
        self.records = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __call__(self, *inputs, **kwargs):
        self.clear()
        with torch.no_grad():
            self.model(*inputs, **kwargs)
        return self.records

    def report(self, show_params=True):
        return _format_report(self.records, show_params=show_params)

    def print_report(self, show_params=True):
        print(self.report(show_params=show_params))


def _format_report(records, show_params=True):
    if not records:
        return '(没有匹配到任何模块，试试 include=None 或 include="all")'

    rows = [
        (i, r['name'] or '(root)', r['type'], str(r['in']), str(r['out']), r['params'])
        for i, r in enumerate(records, 1)
    ]

    sep = '  '
    w_num = max(len(str(r[0])) for r in rows)
    w_name = max(len(r[1]) for r in rows)
    w_type = max(len(r[2]) for r in rows)
    w_in = max(len(r[3]) for r in rows)
    w_out = max(len(r[4]) for r in rows)
    w_par = max(len(format(r[5], ',')) for r in rows) if show_params else 0

    n_sep = 3 + (1 if show_params else 0)
    total = w_num + w_name + w_type + w_in + w_out + 4 + (w_par if show_params else 0) + len(sep) * n_sep

    lines = ['=' * total, '张量数据流追踪（按前向执行顺序）', '=' * total]

    header = (
        '#'.rjust(w_num) + sep
        + '模块'.ljust(w_name) + sep
        + '类型'.ljust(w_type) + sep
        + '输入 -> 输出'.ljust(w_in + w_out + 4)
    )
    if show_params:
        header += sep + '参数量'.rjust(w_par)
    lines.append(header)
    lines.append('-' * total)

    for num, name, typ, in_s, out_s, par in rows:
        row = (
            str(num).rjust(w_num) + sep
            + name.ljust(w_name) + sep
            + typ.ljust(w_type) + sep
            + (in_s + ' -> ' + out_s).ljust(w_in + w_out + 4)
        )
        if show_params:
            row += sep + format(par, ',').rjust(w_par)
        lines.append(row)

    lines.append('=' * total)
    return '\n'.join(lines)


def trace(model, input_shape=(1, 3, 256, 256), input_tensor=None, device='cpu',
          include=None, exclude=None, max_depth=None, show_params=True):
    """一次性追踪：构建假输入 -> 前向 -> 打印表格 -> 返回记录列表。"""
    model = model.to(device).eval()
    if input_tensor is None:
        input_tensor = torch.randn(*input_shape)
    input_tensor = input_tensor.to(device)

    tracer = TensorTracer(model, include=include, exclude=exclude, max_depth=max_depth)
    tracer(input_tensor)
    tracer.print_report(show_params=show_params)
    tracer.remove()
    return tracer.records


def trace_from_config(cfg, input_shape=(1, 3, 256, 256), device='cpu',
                      include=None, exclude=None, max_depth=None, show_params=True):
    """从配置文件（路径或字典）构建 model 段并追踪数据流。"""
    if isinstance(cfg, str):
        from dlkit.utils.config import load_config
        cfg = load_config(cfg)

    import dlkit.models  # noqa: F401  触发模型注册

    from dlkit.registry import build_from_cfg

    model = build_from_cfg(cfg['model'])
    return trace(
        model, input_shape=input_shape, device=device,
        include=include, exclude=exclude, max_depth=max_depth, show_params=show_params,
    )
