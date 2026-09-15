# Git 使用说明（新手版）

这份教程针对当前的 `BlueprintDL` 项目编写。Git 用来记录代码历史、创建分支、合并修改，以及和 GitHub 等远程仓库同步。

日常开发最常用的流程：

```powershell
git status
git add .
git commit -m "描述这次修改"
git push
```

## 1. Git、GitHub 和仓库

- Git：安装在本机上的版本控制工具。
- 仓库：项目目录中的 `.git` 目录，保存提交历史和分支信息。
- GitHub：远程代码托管网站，本项目的远程仓库名称是 `origin`。
- 提交（commit）：某一时刻项目文件的可追踪快照。
- 分支（branch）：在一条历史线上独立开发的指针。

Git 不会自动记录所有文件。当前项目的 `.gitignore` 已经忽略虚拟环境、缓存、数据、训练结果和模型权重等不适合提交的内容。

## 2. 第一次配置

确认 Git 已安装：

```powershell
git --version
```

配置提交时显示的用户名和邮箱：

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

查看配置：

```powershell
git config --global --list
```

进入本项目：

```powershell
cd "A:\从0手搓深度学习项目"
```

## 3. 查看项目状态

```powershell
git status
git branch --show-current
git remote -v
```

`git status` 中常见的状态：

| 状态 | 含义 |
| --- | --- |
| `?? file` | 新文件，还没有被 Git 跟踪 |
| `M file` | 已跟踪文件被修改 |
| `A file` | 文件已添加到暂存区 |
| `D file` | 文件被删除 |

查看未暂存的修改：

```powershell
git diff
```

查看已经放入暂存区的修改：

```powershell
git diff --staged
```

## 4. 提交一次修改

把文件放入暂存区：

```powershell
git add README.md
```

添加当前目录下所有未忽略的修改：

```powershell
git add .
```

添加后检查：

```powershell
git status
git diff --staged
```

创建提交：

```powershell
git commit -m "迁移到 uv 管理依赖"
```

查看最近提交：

```powershell
git log --oneline -5
```

## 5. 本项目哪些文件应该提交

通常应该提交：

- `dlkit/`、`tools/`、`tests/` 中的源代码；
- `configs/` 中的配置文件；
- README、教程和说明文档；
- `pyproject.toml`、`uv.lock` 和 `.python-version`；
- `.gitignore`。

通常不应该提交：

- `.venv/`、`__pycache__/` 和 `.pytest_cache/`；
- `data/` 和 `runs/` 中生成的数据、日志和训练结果；
- `*.pth`、`*.pt`、`*.onnx` 模型文件；
- 密钥、密码、Token 和个人配置。

确认文件是否被忽略：

```powershell
git check-ignore -v .venv
git check-ignore -v data
git check-ignore -v runs
```

本项目迁移到 uv 后，`pyproject.toml`、`uv.lock` 和 `.python-version` 是项目环境的一部分，应该和代码一起提交；`.venv` 只是本机环境，不要提交。

## 6. 分支开发

创建功能分支：

```powershell
git switch -c feature/add-git-guide
```

查看分支：

```powershell
git branch
```

切换分支：

```powershell
git switch main
```

删除已经合并的本地分支：

```powershell
git branch -d feature/add-git-guide
```

如果分支尚未合并，Git 会阻止删除。确认不需要分支内容后，才使用 `git branch -D feature/add-git-guide`。强制删除可能丢失未保存的提交。

## 7. 合并分支

先切换到接收修改的分支，再合并目标分支：

```powershell
git switch main
git pull --ff-only origin main
git merge feature/add-git-guide
```

如果出现冲突：

1. 执行 `git status` 查看冲突文件；
2. 打开文件，处理 `<<<<<<<`、`=======`、`>>>>>>>` 标记；
3. 保存后执行 `git add 冲突文件`；
4. 执行 `git commit` 完成合并。

如果决定取消本次合并：

```powershell
git merge --abort
```

## 8. 和远程仓库同步

只下载远程信息，不修改当前分支：

```powershell
git fetch origin
```

拉取并合并远程 `main`：

```powershell
git pull --ff-only origin main
```

第一次推送新分支：

```powershell
git push -u origin feature/add-git-guide
```

之后可以直接推送：

```powershell
git push
```

推送 `main`：

```powershell
git switch main
git push origin main
```

当前项目使用 SSH 远程地址。如果 SSH 尚未配置，需要先在 GitHub 账户中添加 SSH 公钥。

## 9. 撤销操作

只撤销暂存、保留文件修改：

```powershell
git restore --staged README.md
```

丢弃尚未提交的文件修改：

```powershell
git restore README.md
```

这个操作会删除该文件未提交的修改，执行前要确认内容不再需要。

修改最近一次提交：

```powershell
git add missing_file.py
git commit --amend --no-edit
```

已经推送到远程的提交不建议随意 amend，因为它会改写提交历史。

撤销已经共享的提交，优先使用：

```powershell
git revert <commit-id>
```

它会创建一个新提交来抵消旧提交，不会改写公共历史。

## 10. 查看历史和定位问题

```powershell
git log --oneline --decorate --graph --all
```

查看某次提交：

```powershell
git show <commit-id>
```

查看某个文件的历史：

```powershell
git log --oneline -- dlkit/engine/trainer.py
```

查看两次提交之间的差异：

```powershell
git diff <old-commit> <new-commit>
```

查找某行最后由谁修改：

```powershell
git blame dlkit/engine/trainer.py
```

## 11. 和 uv 配合使用

修改代码并新增依赖时：

```powershell
git switch -c feature/my-change
uv add package-name
uv run pytest
git status
git add dlkit pyproject.toml uv.lock
git commit -m "添加新功能"
git push -u origin feature/my-change
```

如果只是修改代码，不需要修改依赖：

```powershell
uv run pytest
git add dlkit tests
git commit -m "修复模型问题"
```

切换到其他人的修改后：

```powershell
git pull --ff-only origin main
uv sync
uv run pytest
```

## 12. 新手容易犯的错误

### 忘记先看状态

每次提交前先执行：

```powershell
git status
git diff
git diff --staged
```

### 把所有文件都提交

不要提交 `.venv`、数据、训练输出、模型权重或密钥。`.gitignore` 只能忽略未被跟踪的文件；已经被 Git 跟踪的大文件不会因为后来加入 `.gitignore` 就自动移除。

### 在错误的分支上开发

开始工作前先执行：

```powershell
git branch --show-current
git status
```

### 直接使用 `git push --force`

强制推送可能覆盖远程提交。除非明确知道自己在做什么，否则不要使用；确实需要时优先考虑 `git push --force-with-lease`，并先确认远程状态。

## 13. 可以照抄的日常模板

```powershell
cd "A:\从0手搓深度学习项目"
git switch main
git pull --ff-only origin main
git switch -c feature/my-change

# 修改代码或依赖
uv add package-name
uv run pytest

# 检查并提交
git status
git add .
git commit -m "描述本次修改"

# 推送分支
git push -u origin feature/my-change
```

## 参考资料

- [Git 官方文档](https://git-scm.com/doc)
- [Pro Git 中文版](https://git-scm.com/book/zh/v2)
- [GitHub 文档](https://docs.github.com/)
