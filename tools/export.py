import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import dlkit.models  # noqa: F401
from dlkit.utils.config import load_config
from dlkit.registry import build_from_cfg
from dlkit.utils.checkpoint import load_checkpoint
from dlkit.deploy.export import export_onnx, export_torchscript


def main():
    parser = argparse.ArgumentParser(description='Export a trained model')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--format', choices=['onnx', 'torchscript'], default='onnx')
    parser.add_argument('--input-size', nargs=2, type=int, default=[128, 128], metavar=('H', 'W'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_from_cfg(cfg['model'])
    load_checkpoint(args.checkpoint, model=model, device='cpu')

    if args.format == 'onnx':
        export_onnx(model, args.input_size, args.output, device=args.device)
    else:
        export_torchscript(model, args.input_size, args.output, device=args.device)
    print('Exported %s to %s' % (args.format, args.output))


if __name__ == '__main__':
    main()
