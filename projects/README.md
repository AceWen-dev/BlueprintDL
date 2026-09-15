# BlueprintDL Projects

这个目录存放在 BlueprintDL 规则下孵化的具体深度学习项目。每个一级子目录都是独立 Python 项目，并拥有自己的 `project.yaml`、源码、配置、测试和文档。

项目可以使用 `dlkit` 的公共能力，但 `dlkit` 不得依赖这里的任何项目，项目之间也不得互相导入内部代码。

项目组件通过显式 bootstrap 注册，并使用 `provider=<project_id>` 记录归属。使用 Forge 的 `list`、`inspect` 和 `audit` 命令可以追踪项目及组件来源。

完整规则见 [项目锻造标准](../standards/PROJECT_STANDARD.md)。
