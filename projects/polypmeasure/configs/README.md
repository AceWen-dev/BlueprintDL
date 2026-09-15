# PolypMeasure 配置目录

项目的训练、评估、推理和导出配置放在这里。配置只保存注册名称和可序列化参数，例如：

```yaml
model:
  type: PolypNet
  params:
    num_classes: 2

optimizer:
  type: AdamW
  params:
    lr: 0.0003

runtime:
  device: cuda
  amp: true
```

配置不保存 `model.parameters()`、optimizer 实例、device 对象、logger、凭据或本机 CUDA 路径。这些运行时依赖由项目 Composition Root、Builder 或 Pipeline setup 注入。

当前目录尚未提供业务配置；需要先明确 PolypMeasure 的数据协议、模型与测量流程，再添加可以被测试和项目入口实际使用的配置。
