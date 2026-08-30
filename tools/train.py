import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from dlkit.utils.config import load_config, apply_overrides
from dlkit.engine.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train a model from a config file')
    parser.add_argument('--config', required=True, help='path to yaml config')
    parser.add_argument('--work-dir', default=None, help='override output directory')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    parser.add_argument('--device', default=None, help='cpu, cuda or auto')
    parser.add_argument('--opts', nargs='*', default=[], help='override config key=value, e.g. train.epochs=10')
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.opts)

    trainer = Trainer(cfg, work_dir=args.work_dir, device=args.device, resume_from=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
