# Prior art：研究者为什么自建实验系统

> 调研时间：2026-10。目的：给 BlueprintDL 的 catalog、Recipe、extensions 和实验治理找设计先例；不是功能承诺，也不代表这些项目都适合直接引入。

## 一句话结论

“实验太多、脚本太乱、数据划分记不住、图找不到”不是个人问题，而是研究工程的常见失败模式。过去十年，研究者们反复收敛到同一组设计：**配置 + 注册表 + 单一入口 + hooks + 扩展目录 + run 记录**；而“实验目录 / 数据版本”通常被单独做成一层，不塞进 Trainer。

## 1. 实验室级统一框架

这些项目的起点几乎都是“每个子项目都有一套 train.py”。

- **MMEngine / OpenMMLab**：Config + Registry + Runner + Hook + Recorder。官方迁移文档原话：OpenMMLab 1.x 里“每个仓库都自己搭 Runner，造成大量冗余代码”。这是和 BlueprintDL 六契约最接近的成熟实现。
  - https://github.com/open-mmlab/mmengine
  - https://mmengine.readthedocs.io/en/latest/design/runner.html
- **Detectron2（FAIR）**：定位是 research platform；YAML config + registry + hooks，`projects/` 是库外扩展的标准位置。
  - https://github.com/facebookresearch/detectron2
  - https://detectron2.readthedocs.io/en/latest/tutorials/write-models.html
- **fairseq2（Meta）**：官方说明 fairseq v1 是 monolithic framework，项目代码互相纠缠；fairseq2 从零重写，让研究者独立拥有项目代码，并提供 recipes + 结构化配置 + extension 机制。
  - https://github.com/facebookresearch/fairseq2
- **PyTorch Lightning（William Falcon）**：2015 年在 Columbia 开始写，2019 年在 NYU + FAIR 开源；Trainer + callbacks + loggers + checkpointing，动机就是减少反复重写训练循环。
  - https://lightning.ai/about
- **torchtitan / torchtune（Meta PyTorch）**：torchtitan 是单一 `train.py` + 配置 + recipes + `experiments/`；torchtune 明确把 recipes 作为主要入口，并声明 recipe “不是 monolithic trainer、不是 generalized entry point”。
  - https://github.com/pytorch/torchtitan
  - https://github.com/meta-pytorch/torchtune
  - https://meta-pytorch.org/torchtune/main/basics/custom_components.html
- **torchvision references**：每个任务一个标准 `train.py` + preset 配置 + 公布结果，官方 baseline 的做法。
  - https://github.com/pytorch/vision/tree/main/references

## 2. 个人研究者的实验系统

这些更接近“一个人面对一堆实验”的场景。

- **timm（Ross Wightman）**：一个 `train.py` + 参数 + `results/*.csv` + README 大结果表；大量训练 run 靠 CSV 索引。极简版“实验目录”。
  - https://github.com/rwightman/timm
  - https://github.com/rwightman/timm/blob/main/results/README.md
- **Sacred（Klaus Greff 等，IDSIA）**：论文标题就是 *The Sacred Infrastructure for Computational Research*；目标是管理配置、复现结果和实验记账，配套 Sacredboard 看板。
  - https://github.com/IDSIA/sacred
  - https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/17-SciPy-Sacred.pdf
- **LabML（Varuna Jayasiri）**：副标题就是 *A library to organize machine learning experiments*；两行接入，记录 git commit、配置、超参和指标。
  - https://github.com/labmlai/labml
- **CleanRL（Costa Huang 等）**：反方向哲学——single-file implementation + W&B 实验管理 + benchmark 网站；JMLR 2022 论文解释为什么单文件更适合研究。
  - https://github.com/vwxyzjn/cleanrl
  - https://jmlr.org/papers/v23/21-1342.html
- **LitGPT（Lightning AI，Sebastian Raschka 等）**：minimal、readable、YAML config + recipes + 单一 CLI。
  - https://github.com/Lightning-AI/litgpt
- **LLaMA-Factory（hiyouga）**：把大量微调脚本统一成一个 YAML + CLI + WebUI，是中文社区最典型的“我不想再写 train_xxx.py”案例。
  - https://github.com/hiyouga/LLaMA-Factory
- **Guild AI（Garrett Smith）**：单人开发的实验追踪工具，不改代码即可记录 run；强调“Git + SQLite + Make 式的 lab notebook”。
  - https://github.com/guildai/guildai
- **DVC（Dmitry Petrov）**：起因原话是数据科学家在文件夹之间挣扎，把文件命名成 `data_final_v3_really_final.zip`，还复现不了实验。数据/模型版本控制 + 实验管理。
  - https://github.com/treeverse/dvc
  - https://dvc.org/blog/a-shared-vision-for-the-future-of-dvc

## 3. 领域系统（医疗 / 语音）

- **nnU-Net（DKFZ）**：数据集指纹 + 固定/规则参数 + 5 折交叉验证，23 个公开数据集不用手写脚本。论文明确指出手工为每个数据集定制流程“容易出错、不可扩展，成败几乎完全取决于实验者经验”。这直接对应“数据集多、划分多”的混乱。
  - https://github.com/MIC-DKFZ/nnUNet
  - https://www.nature.com/articles/s41592-020-01008-z
- **MONAI / MONAI Bundle**：Bundle 把权重、训练配置、metadata、推理代码打包成一个可复现单元；Model Zoo 已托管几十个模型。医疗影像界的“run.json + 权重 + 配置”标准。
  - https://project-monai.github.io/model-zoo.html
  - https://monai.readthedocs.io/en/stable/bundle_intro.html
- **ESPnet / SpeechBrain**：ESPnet 每个语料一个 `run.sh`，从下载到打分，所有任务共享同一套 recipe 结构；SpeechBrain 用 HyperPyYAML + Brain 类 + recipes，并公开实验日志和曲线。**Recipe 这个词就是从这类系统来的**。
  - https://github.com/espnet/espnet
  - https://github.com/speechbrain/speechbrain
- **OpenML**：把 datasets / tasks / flows / runs 统一登记到线上，学术版的实验索引。
  - https://www.openml.org
  - https://arxiv.org/pdf/1407.7722

## 4. 索引 / 追踪层

这些工具证明：实验目录和训练框架应该分开。

- **MLflow**：Tracking + Model Registry + lineage，“哪个实验产出了哪个模型”是一等功能。
  - https://mlflow.org/docs/latest/ml/tracking/
- **Aim**：明确面向“10,000s of training runs”；README 直接批评 TensorBoard 需要把参数写进 run 名字才能搜索/聚合。
  - https://github.com/aimhubio/aim
- **XManager（Google DeepMind）**：packaging、running、tracking ML experiments；Google 内部管理大量实验的平台开源版。
  - https://github.com/google-deepmind/xmanager
- **Hydra / Optuna**：配置组合与调参。Hydra 是 Omry Yadan 2019 年在 Facebook 为配置爆炸开始写的；Optuna 是 Preferred Networks 为自己内部调参实验做的。
  - https://hydra.cc
  - https://github.com/optuna/optuna
- **ML Reproducibility Checklist（Joelle Pineau）**：被 NeurIPS 采纳，强制写清数据划分、超参范围、运行次数、误差棒。
  - https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf

## 5. 收敛设计对照

| BlueprintDL 契约 | 先例 |
| --- | --- |
| 配置结构（type + params） | MMEngine Config、Detectron2 CfgNode、Hydra、torchtitan dataclass |
| Registry | MMEngine / Detectron2 registries、fairseq2 extension |
| Run 身份 | MLflow run、XManager experiment、Guild run、LabML experiment |
| Trainer / Hook | Lightning Trainer + callbacks、MMEngine Runner + Hook、SpeechBrain Brain |
| extensions / imports | Detectron2 `projects/`、torchtitan `experiments/`、fairseq2 extensions |
| 实验工具 | timm 的 results.csv、ESPnet RESULTS.md、Aim/MLflow/DVC |
| Recipe | ESPnet/SpeechBrain recipes、torchtune recipes、fairseq2 recipes |
| 数据/划分 | DVC、nnU-Net fingerprints、MONAI Bundle、OpenML |

## 6. 反面教训

- **OpenAI Baselines**：想做统一 baseline，但每个算法独立实现、序列化 API 也未统一（README 自己承认），最后被 Stable-Baselines3 / CleanRL 取代。**统一接口比统一算法难**。
  - https://github.com/openai/baselines
- **fairseq v1**：脚本各自为战、项目代码互相纠缠，最终 Meta 选择从零重写 fairseq2。
- **框架变重**：MMDetection 的配置层级、Lightning 的抽象层，都是“统一”带来的真实成本。BlueprintDL 保持 6 契约 + 项目扩展，是在避免重演。
- **单文件 vs 框架**：CleanRL 选择“每个算法一个文件”，MMEngine 选择“统一 Runner”；两者都能成功，关键是一致性和可读性，而不是抽象层数。

## 7. 对 BlueprintDL 的边界

1. **Catalog 是索引层**，不是 Trainer 功能；参考 MLflow / Aim / XManager，而不是把查询逻辑塞进训练循环。
2. **Dataset / Split Registry 是数据层**，参考 DVC / nnU-Net fingerprint / MONAI Bundle；`dataset_meta.json` 只是起点，后续需要 split manifest 和稳定 ID。
3. **Recipe 是一等公民**：ESPnet / SpeechBrain / torchtune 已经证明，比“每个想法一个脚本”更可维护。
4. **不要追通用大框架的体量**：BlueprintDL 的差异化是“个人可控 + 领域实验 + 可读性”，不是功能数量。
5. **YOLO 厂商训练器留在领域项目**：它是训练器，不是实验操作系统；BlueprintDL 只接推理/评测和权重资产。
