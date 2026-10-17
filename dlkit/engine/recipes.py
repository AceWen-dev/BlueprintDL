"""训练流程插件(Recipe)。

Recipe 用来承载"流程型想法"——它改变的是训练怎么进行,而不是某个组件是什么:

- 一致性训练 / EMA 教师
- 自蒸馏、伪标签、交替优化
- 自定义 step 逻辑、训练中动态调整组件

一个 Recipe 只需实现 ``attach(trainer)``:注册 hook、准备辅助模块
(例如把 EMA teacher 挂到 ``trainer.ema_model``),必要时在 trainer 上暴露
额外对象。配置示例::

    recipe:
      type: EmaTeacher
      params:
        decay: 0.999
        validate_teacher: true

原则:流程变化写成可命名、可配置、可复用的 Recipe,而不是复制一份训练脚本。
"""

from __future__ import annotations

import copy

import torch

from dlkit.registry import RECIPES


class BaseRecipe:
    """Recipe 协议:attach(trainer) 在模型/优化器构建完成后调用。"""

    def attach(self, trainer):
        raise NotImplementedError


@RECIPES.register(
    tags=('ema', 'semi-supervised', 'template'),
    description='EMA 教师模型:逐步跟随学生权重,验证时可选切换为教师权重',
)
class EmaTeacher(BaseRecipe):
    """指数滑动平均教师(Mean Teacher / 半监督与自蒸馏的常见基础设施)。

    行为:
    - attach 时深拷贝当前学生模型作为教师,并挂在 ``trainer.ema_model`` 上;
    - 每个训练 step 后按 ``decay`` 更新教师权重(前 warmup_steps 步直接拷贝);
    - ``validate_teacher=True`` 时,验证阶段临时切换成教师权重(EMA 评估),
      验证后恢复学生权重,不影响 optimizer/checkpoint。

    Loss/Recipe 可以通过 ``trainer.ema_model`` 取教师做一致性约束。
    """

    def __init__(self, decay=0.999, warmup_steps=0, validate_teacher=True):
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError('decay must be within [0, 1]')
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.validate_teacher = bool(validate_teacher)
        self.teacher = None
        self._step = 0
        self._student_backup = None

    def attach(self, trainer):
        self.trainer = trainer
        self.teacher = copy.deepcopy(trainer.model)
        self.teacher.to(trainer.device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        trainer.ema_model = self.teacher
        trainer.register_hook('after_train_step', self._after_step)
        if self.validate_teacher:
            trainer.register_hook('before_validation', self._swap_in_teacher)
            trainer.register_hook('after_validation', self._swap_out_teacher)
        trainer.logger.info(
            'EmaTeacher attached (decay=%.4f, warmup=%d, validate_teacher=%s)',
            self.decay, self.warmup_steps, self.validate_teacher,
        )

    def _after_step(self, trainer, **kwargs):
        self._step += 1
        decay = 1.0 if self._step <= self.warmup_steps else self.decay
        self._update(decay)

    def _update(self, decay):
        with torch.no_grad():
            teacher_state = self.teacher.state_dict()
            student_state = self.trainer.model.state_dict()
            for key, teacher_value in teacher_state.items():
                student_value = student_state[key]
                if teacher_value.dtype.is_floating_point:
                    teacher_value.mul_(decay).add_(student_value.detach(), alpha=1.0 - decay)
                else:
                    teacher_value.copy_(student_value)

    def _swap_in_teacher(self, trainer, **kwargs):
        self._student_backup = {
            key: value.detach().clone() for key, value in trainer.model.state_dict().items()
        }
        trainer.model.load_state_dict(self.teacher.state_dict())

    def _swap_out_teacher(self, trainer, **kwargs):
        if self._student_backup is not None:
            trainer.model.load_state_dict(self._student_backup)
            self._student_backup = None


__all__ = ['BaseRecipe', 'EmaTeacher']
