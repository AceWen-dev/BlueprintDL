"""BlueprintDL 可视化组件包(VISUALIZERS Registry 的实现)。

导入本包即完成所有内置可视化组件的注册。协议约定:

    visualizer.render(ctx: RunContext) -> list[str]  # 写入的文件路径

模型/数据依赖型组件在 ctx.model 或 ctx.val_loader 缺失时抛 RuntimeError,
由工具层(tools/visualize.py)决定跳过或降级。
"""

# noqa: F401  组件模块注册需要副作用导入
from dlkit.viz.curves import TrainingCurves  # noqa: F401
from dlkit.viz.seg import SegOverlay  # noqa: F401
from dlkit.viz.classification import ConfusionMatrix  # noqa: F401
from dlkit.viz.det import DetBoxes  # noqa: F401
from dlkit.viz.depth import DepthColormap  # noqa: F401
from dlkit.viz.context import RunContext, build_run_context

__all__ = ['RunContext', 'build_run_context']
