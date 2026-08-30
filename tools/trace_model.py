import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import dlkit.models  # noqa: F401  触发模型注册

from dlkit.registry import build_from_cfg
from dlkit.utils.config import load_config, apply_overrides
from dlkit.utils.tracer import trace


def main():
    parser = argparse.ArgumentParser(description='可视化模型中张量的数据流形状')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--input-size', nargs='+', type=int, default=[3, 256, 256],
                        help='输入形状 C H W（默认 3 256 256），也可给 B C H W')
    parser.add_argument('--device', default='cpu', help='cpu 或 cuda（默认 cpu）')
    parser.add_argument('--include', default=None,
                        help='只追踪特定层: all / conv / bn / pool / activation / linear')
    parser.add_argument('--max-depth', type=int, default=None, help='最大嵌套层数')
    parser.add_argument('--no-params', action='store_true', help='不显示参数量')
    parser.add_argument('--opts', nargs='*', default=[], help='覆盖配置 key=value')
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.opts)

    model = build_from_cfg(cfg['model'])

    input_shape = tuple(args.input_size)
    if len(input_shape) == 3:
        input_shape = (1,) + input_shape

    include = None if args.include in (None, 'all') else args.include

    trace(
        model,
        input_shape=input_shape,
        device=args.device,
        include=include,
        max_depth=args.max_depth,
        show_params=not args.no_params,
    )


if __name__ == '__main__':
    main()
