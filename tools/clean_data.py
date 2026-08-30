import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from dlkit.utils.config import load_config
from dlkit.data.cleaning import run_cleaning


def main():
    parser = argparse.ArgumentParser(description='Clean a dataset according to a config')
    parser.add_argument('--config', required=True, help='config with a "clean" section')
    args = parser.parse_args()

    cfg = load_config(args.config)
    clean_cfg = cfg.get('clean')
    if not clean_cfg:
        raise SystemExit('Config has no "clean" section')

    summary = run_cleaning(clean_cfg)
    print('Found %d issues, report written to %s'
          % (summary['total_issues'], clean_cfg.get('output_dir', 'runs/clean_report')))


if __name__ == '__main__':
    main()
