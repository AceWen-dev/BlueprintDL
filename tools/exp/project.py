"""项目生命周期:创建/查看/收尾实验项目。

    uv run python tools/exp/project.py new myseg --display-name "Medical Seg" --task segmentation
    uv run python tools/exp/project.py list
    uv run python tools/exp/project.py status myseg
    uv run python tools/exp/project.py archive myseg

约定:
- 项目骨架在 ``projects/<id>/``:project.yaml + configs/ + reports/ + data/README;
- ``archive`` 只生成收尾报告(closure + EXPERIMENTS.md/CSV/曲线图)并把状态置为
  archived;**runs 产物全部保留**,不移动不删除——随时可以 revive 重新对比。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import datetime
import re
from pathlib import Path

import yaml

PROJECTS_DIR_DEFAULT = 'projects'
RUNS_ROOT_DEFAULT = 'runs'
PROJECT_ID_RE = re.compile(r'^[a-z][a-z0-9_-]*$')

_SKELETON_README = {
    'configs': (
        '# configs\n\n'
        '一个实验一份 YAML。从 `_template.yaml` 复制后改名:\n\n'
        '- `project` 字段保持本项目的 id 不变;`name` 改成新实验名(决定 runs 子目录);\n'
        '- 训练:`uv run python tools/train.py --config projects/<id>/configs/<experiment>.yaml`\n'
        '- 对比:`uv run python tools/exp/compare.py --project <id>`\n'
    ),
    'reports': (
        '# reports\n\n'
        '`tools/exp/compare.py` 与 `tools/exp/project.py archive` 的产物(EXPERIMENTS.md、'
        'closure_*.md、曲线图)落在这里。\n'
    ),
    'data': (
        '# data\n\n'
        '建议把原始数据与本项目的 `dataset_meta.json`(由 `tools/data/prepare_data.py` 生成)放在附近,'
        '例如 `data/<id>/`。数据不属于 Git。\n'
    ),
    'extensions': (
        '# extensions(项目专属扩展)\n\n'
        '这里放"只属于本项目"的代码,不要污染 `dlkit/`,也不要新建训练入口脚本:\n\n'
        '- 新 Dataset/Loss/Model/Metric/Transform:写在 .py 里并用对应 Registry 注册;\n'
        '- 流程型想法(EMA 教师、一致性训练、自蒸馏、交替优化):写成 Recipe,注册到 RECIPES;\n'
        '- 配置中用 `imports: [projects/<id>/extensions]` 声明,训练/评估/推理会自动加载;\n'
        '- 一次性探索放 `scratch/`(gitignore),不要留在 extensions/。\n'
    ),
}

_TASK_TEMPLATES = {
    'segmentation': 'unet_toy.yaml',
    'classification': 'cls_toy.yaml',
    'detection': 'det_toy.yaml',
    'depth': 'depth_toy.yaml',
}


# ---------- 项目文档(project.yaml) ----------


def valid_project_id(project_id: str) -> bool:
    return bool(PROJECT_ID_RE.match(project_id or ''))


def project_path(projects_dir: str | Path, project_id: str) -> Path:
    return Path(projects_dir) / project_id


def load_project(path: Path) -> dict | None:
    meta_path = Path(path) / 'project.yaml'
    if not meta_path.is_file():
        return None
    with meta_path.open('r', encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}
    return doc if isinstance(doc, dict) else None


def save_project(path: Path, doc: dict) -> Path:
    meta_path = Path(path) / 'project.yaml'
    meta_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return meta_path


def create_project(
    projects_dir: str | Path,
    project_id: str,
    *,
    display_name: str | None = None,
    task: str = 'segmentation',
) -> Path:
    if not valid_project_id(project_id):
        raise ValueError('invalid project id %r (use lowercase kebab-case like polyp-seg)' % project_id)
    root = project_path(projects_dir, project_id)
    if root.exists():
        raise FileExistsError('project already exists: %s' % root)

    (root / 'configs').mkdir(parents=True)
    (root / 'reports').mkdir()
    (root / 'data').mkdir()
    (root / 'extensions').mkdir()

    save_project(root, {
        'schema_version': 1,
        'id': project_id,
        'display_name': display_name or project_id,
        'task': task,
        'status': 'active',
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'runs_root': RUNS_ROOT_DEFAULT,
        'data_root': 'data/%s' % project_id,
    })

    for sub, text in _SKELETON_README.items():
        (root / sub / 'README.md').write_text(text, encoding='utf-8')
    (root / 'configs' / '_template.yaml').write_text(
        _render_template(project_id, task), encoding='utf-8')
    (root / 'reports' / '.gitkeep').touch()
    return root


def _render_template(project_id: str, task: str) -> str:
    """从框架 toy 配置生成项目模板:唯一改动是 project/name 与数据根目录。"""

    source = Path(__file__).resolve().parents[2] / 'configs' / _TASK_TEMPLATES[task]
    cfg = yaml.safe_load(source.read_text(encoding='utf-8')) or {}

    cfg['project'] = project_id
    cfg['name'] = 'template'
    # 项目专属组件/Recipe 的目录:训练、评估、推理都会自动加载
    cfg['imports'] = ['projects/%s/extensions' % project_id]

    data_root = Path('data') / project_id
    data_cfg = cfg.get('data') or {}
    for split in ('train', 'val'):
        params = (data_cfg.get(split) or {}).get('params') or {}
        for key in ('image_dir', 'mask_dir', 'label_dir', 'depth_dir', 'root'):
            if key in params:
                old_parts = Path(str(params[key])).parts
                if split in old_parts:
                    index = old_parts.index(split)
                    params[key] = data_root.joinpath(*old_parts[index:]).as_posix()
                else:
                    params[key] = (data_root / split).as_posix()
    loader = data_cfg.get('loader') or {}
    loader['num_workers'] = 0
    loader.setdefault('pin_memory', False)

    header = (
        '# 模板配置(project.py new 生成)。复制为 <experiment>.yaml 后使用:\n'
        '# - project 保持 %s 不变;name 改成你的实验名\n'
        '# - 依据 data/prepare_data 的 dataset_meta.json 核对 num_classes / 数据路径\n'
        '# - 项目专属组件与 Recipe 放 extensions/,由 imports 自动加载\n'
        '# - 原始参考配置: configs/%s\n\n' % (project_id, _TASK_TEMPLATES[task])
    )
    return header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def _runs_summary(runs_root: str | Path, project_id: str) -> dict:
    from dlkit.utils.runs import iter_run_dirs

    run_dirs = iter_run_dirs(runs_root, project=project_id)
    experiments = sorted({d.parent.name for d in run_dirs})
    latest = run_dirs[-1] if run_dirs else None
    return {
        'num_runs': len(run_dirs),
        'experiments': experiments,
        'latest': str(latest) if latest else None,
    }


def summarize_projects(projects_dir: str | Path, runs_root: str | Path = RUNS_ROOT_DEFAULT) -> list[dict]:
    root = Path(projects_dir)
    if not root.is_dir():
        return []
    rows = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        doc = load_project(child)
        if doc is None:
            continue
        row = dict(doc)
        row['_path'] = child
        try:
            row['runs'] = _runs_summary(runs_root, doc.get('id') or child.name)
        except Exception as exc:  # runs 扫描失败不阻塞 list
            row['runs'] = {'error': str(exc)}
        rows.append(row)
    return rows


# ---------- 收尾(archive) ----------


def set_project_status(path: Path, status: str) -> dict:
    doc = load_project(path)
    if doc is None:
        raise FileNotFoundError('project.yaml not found under %s' % path)
    doc['status'] = status
    if status == 'archived':
        doc['archived_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    save_project(path, doc)
    return doc


def write_closure_report(
    project_root: str | Path,
    doc: dict,
    rows: list[dict],
    metric: str | None,
) -> list[Path]:
    """生成 closure_<时间>.md + EXPERIMENTS 产物,返回写出的文件列表。"""

    from dlkit.utils.experiments import write_outputs

    reports_dir = Path(project_root) / 'reports'
    outputs = list(write_outputs(rows, doc.get('id') or Path(project_root).name, reports_dir, metric))
    closure = reports_dir / ('closure_%s.md' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    closure.write_text(_closure_markdown(doc, rows, metric), encoding='utf-8')
    return [closure] + outputs


def _closure_markdown(doc: dict, rows: list[dict], metric: str | None) -> str:
    num_runs = len(rows)
    experiments = sorted({str(row['experiment']) for row in rows})
    lines = [
        '# Project closure: %s' % doc.get('display_name', doc.get('id', '?')),
        '',
        '- project id: `%s`' % doc.get('id', '?'),
        '- task: %s | status: %s' % (doc.get('task', '?'), doc.get('status', '?')),
        '- archived at: %s' % datetime.datetime.now().isoformat(timespec='seconds'),
        '- summarized runs: %d | experiments: %s' % (num_runs, ', '.join(experiments) or '-'),
        '- preferred metric: %s' % (metric or '-'),
        '',
        '## Best result per experiment',
        '',
        '| experiment | run_id | best metric | value | epoch | device |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        best = row.get('best')
        if not best:
            continue
        lines.append('| %s | %s | %s | %.4f | %s | %s |' % (
            row['experiment'], row['run_id'], best.get('name'),
            float(best.get('value') or 0.0), best.get('epoch'), row.get('device', '-')))
    if not any(row.get('best') for row in rows):
        lines.append('| - | - | (no best metric recorded) | - | - | - |')
    lines += [
        '',
        '## Verify before closing',
        '',
        '- [ ] 数据档案(dataset_meta.json)与代码状态已提交/归档;',
        '- [ ] 关键 run 的 checkpoint 与 vis/ 结果仍保留在 `runs/` 下(收尾不删不改);',
        '- [ ] `scratch/` 一次性脚本已清理,或已提升为 `extensions/` 组件/Recipe;',
        '- [ ] 下一步计划写进本文件的附加说明区。',
        '',
        'runs 目录保持原样,随时可用 `tools/exp/compare.py --project %s` 重新汇总。' % doc.get('id', '?'),
        '',
    ]
    return '\n'.join(lines)


# ---------- CLI ----------


def build_parser():
    parser = argparse.ArgumentParser(description='Project lifecycle: new/list/status/archive')
    sub = parser.add_subparsers(dest='command')

    def _common(child):
        child.add_argument('--projects-dir', default=PROJECTS_DIR_DEFAULT, help='projects root (default: projects)')
        child.add_argument('--runs-root', default=RUNS_ROOT_DEFAULT, help='runs root (default: runs)')

    new = sub.add_parser('new', help='scaffold a new experiment project')
    new.add_argument('id', help='project id (lowercase kebab-case)')
    new.add_argument('--display-name', default=None)
    new.add_argument('--task', default='segmentation',
                     choices=sorted(_TASK_TEMPLATES.keys()))
    _common(new)

    list_cmd = sub.add_parser('list', help='list projects with run counts')
    _common(list_cmd)
    status = sub.add_parser('status', help='show one project detail')
    status.add_argument('id')
    _common(status)

    archive = sub.add_parser('archive', help='write closure report and mark archived')
    archive.add_argument('id')
    _common(archive)
    return parser


def main():
    args = build_parser().parse_args()
    projects_dir = Path(args.projects_dir)

    if args.command == 'new':
        try:
            root = create_project(
                projects_dir, args.id,
                display_name=args.display_name, task=args.task,
            )
        except (ValueError, FileExistsError) as exc:
            print('ERROR: %s' % exc)
            return 1
        print('Created.  %s' % root)
        print('Next: put data under %s and copy %s to your first experiment config.'
              % (Path('data') / args.id, root / 'configs' / '_template.yaml'))
        return 0

    if args.command == 'list':
        rows = summarize_projects(projects_dir, args.runs_root)
        if not rows:
            print('No projects under %s (use "project.py new <id>")' % projects_dir)
            return 0
        print('%-16s %-14s %-10s %5s %5s  %s' % ('id', 'task', 'status', 'cfgs', 'runs', 'latest run'))
        for row in rows:
            configs = len(list((row['_path'] / 'configs').glob('*.yaml'))) if (row['_path'] / 'configs').is_dir() else 0
            runs = row.get('runs') or {}
            print('%-16s %-14s %-10s %5d %5d  %s' % (
                row.get('id'), row.get('task'), row.get('status'),
                configs, runs.get('num_runs', 0), runs.get('latest') or '-'))
        return 0

    if args.command == 'status':
        root = project_path(projects_dir, args.id)
        doc = load_project(root)
        if doc is None:
            print('ERROR: no project.yaml under %s' % root)
            return 1
        runs = _runs_summary(args.runs_root, doc.get('id') or args.id)
        print('id: %s | task: %s | status: %s' % (doc.get('id'), doc.get('task'), doc.get('status')))
        print('created: %s | archived: %s' % (doc.get('created_at'), doc.get('archived_at') or '-'))
        print('runs: %d across %d experiments' % (runs['num_runs'], len(runs['experiments'])))
        print('latest: %s' % (runs['latest'] or '-'))
        return 0

    if args.command == 'archive':
        root = project_path(projects_dir, args.id)
        doc = load_project(root)
        if doc is None:
            print('ERROR: no project.yaml under %s' % root)
            return 1

        from dlkit.utils.experiments import collect_rows, default_metric

        rows = collect_rows(args.runs_root, doc.get('id') or args.id)
        metric = default_metric(rows)
        doc = set_project_status(root, 'archived')
        outputs = write_closure_report(root, doc, rows, metric)

        print('Runs kept untouched: %d' % len(rows))
        print('Outputs:')
        for path in outputs:
            print('  Wrote %s' % path)
        print('Mark archived: %s -> %s' % (doc.get('id'), doc.get('status')))
        return 0

    print('usage: project.py {new,list,status,archive} (see --help)')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
