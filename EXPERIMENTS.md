# MCDHGN：逐基因元路径权重与 MODIG 思想的消融实验

这套代码为训练服务器准备。本地不执行完整训练，也不要求使用本机创建的环境。将 `model.py`、`fusion.py`、`ablation_config.py`、`run_ablation.py`、`summarize_ablations.py` 和 `tests/` 同步到服务器项目目录，在原有可运行的 PyTorch / DGL 环境中执行下面的命令。

## 1. 借鉴的到底是什么

参考：[MODIG 论文 §2.4，式 11–12](https://academic.oup.com/bioinformatics/article/38/21/4901/6696210)；[作者模型代码](https://github.com/zjupgx/modig/blob/main/modig.py)。

- 图内编码得到每个视图的基因表示 `Z_m`。
- 保留原表示的融合为 `Z'_m = (1-alpha) Z_m + alpha Z_shared_m`。这里的“原表示”是**图编码后的视图表示**，不是原始 48 维多组学输入。
- 最终视图权重根据每个基因计算，而不是先在全体基因上平均。

论文把中间模块称为跨维度信息共享，但公开代码逐个调用 `self_attn(emb)`，其中 `emb` 是 `[N, D]`，注意力矩阵为基因×基因；最终 `Attention` 才接收 `[N, M, D]` 并在 `dim=1` 上归一化。公开实现还共享图编码器；本项目每条元路径有独立编码器。

本次保留 MCDHGN 的独立编码器和单层配置，直接测试逐基因权重；中间模块另设计为“同一基因的 M 个元路径视图之间的信息交互”。它受到 MODIG 思路启发，**不是 MODIG 原版复现**。没有照搬跨全部基因的 N×N 注意力，也没有同时加入两层编码器、LayerNorm、原始特征残差或新损失，以免无法归因。

## 2. 如何得到每个基因自己的元路径权重

设 `z.shape = [N, M, D]`；N 是基因数，M=9 是元路径数，当前 D=256×4=1024。

原版的核心操作：

```python
scores = project(z)                       # [N, M, 1]
beta = softmax(scores.mean(0), dim=0)      # [M, 1]，所有基因共用
gene_features = (beta[None] * z).sum(1)    # [N, D]
```

逐基因版本：

```python
scores = project(z)                       # [N, M, 1]
beta = softmax(scores, dim=1)             # [N, M, 1]，每个基因自己的9个权重
gene_features = (beta * z).sum(1)         # [N, D]
```

两者使用完全相同的评分 MLP，参数量相同。区别仅为是否在基因维度先求均值。每个基因的九个权重之和都为 1。

### 2.1 先把三个维度分清楚

> **先记住一句话：** 原版先问“总体上哪条元路径更有用”，再把同一组权重用于所有基因；逐基因版直接问“对当前这个基因，哪条元路径更有用”。

一个基因经过九个元路径编码器后，会得到九个向量，而不是一个向量。以基因 A 为例，它有 `z[A, PPI]`、`z[A, Pathway]`、`z[A, GO]` 等九份表示；每份当前有 1024 个数。

```text
z[N, M, D]
  N：基因，沿 dim=0 排列
  M：元路径，沿 dim=1 排列，当前为 9
  D：每个表示的特征，沿 dim=2 排列，当前为 1024

project：Linear(D, 128) → Tanh → Linear(128, 1)
scores = project(z)                  # [N, M, 1]
```

`project` 将每个“基因—元路径”的向量变成一个评分。评分可以为正或负，还不是概率；softmax 才把评分转为和为 1 的权重。同一个评分网络用于所有基因和元路径。

### 2.2 原版：先对基因求平均，再算一套共同权重

为了看清操作，假设只有 A、B 两个基因和两条元路径，评分如下。**这些数只用于解释计算，不是本项目的实验结果。**

```text
                 元路径 1     元路径 2
基因 A 的评分        2            0
基因 B 的评分        0            2
```

1. `scores.mean(0)` 沿基因方向求平均，每条元路径留下一个分数。
2. 元路径 1 的平均分为 `(2+0)/2=1`，元路径 2 也为 `(0+2)/2=1`。
3. `softmax([1, 1])=[0.5, 0.5]`，得到所有基因共用的一套权重。
4. 用这套权重分别融合 A、B 自己的向量。

```text
h_A = 0.5 × z[A, 1] + 0.5 × z[A, 2]
h_B = 0.5 × z[B, 1] + 0.5 × z[B, 2]
```

> **共用权重不等于共用表示。** A、B 的输入向量不同，所以融合结果仍然可以不同。原版丢掉的是“ A 更偏向路径 1，而 B 更偏向路径 2 ”的局部评分差异。

这里是**先平均原始评分，再做 softmax**；不是先给每个基因算权重，再把这些权重平均。两种计算通常不等价。

### 2.3 逐基因版：保留评分矩阵，分别在每个基因内部做 softmax

去掉 `mean(0)`，对 `[N, M, 1]` 的 `dim=1` 做 softmax。仍然使用上一节的评分：

```text
基因 A：softmax([2, 0]) = [0.8808, 0.1192]
基因 B：softmax([0, 2]) = [0.1192, 0.8808]

h_A = 0.8808 × z[A, 1] + 0.1192 × z[A, 2]
h_B = 0.1192 × z[B, 1] + 0.8808 × z[B, 2]
```

现在 A 更依赖路径 1，B 更依赖路径 2。这就是“每个基因自己的元路径权重”。实际代码有九条元路径，同样在每一行的九个评分上做 softmax。

### 2.4 对照代码，每一行到底改变了什么

```python
# 两个版本共有：同一份参数、同一个评分网络
scores = self.project(z)                 # [N, M, 1]

# 原版 global
global_scores = scores.mean(dim=0)       # [M, 1]：基因维度被平均掉
global_beta = torch.softmax(global_scores, dim=0)
global_h = (z * global_beta.unsqueeze(0)).sum(dim=1)  # [N, D]

# 逐基因 node
node_beta = torch.softmax(scores, dim=1)  # [N, M, 1]：保留基因维度
node_h = (z * node_beta).sum(dim=1)       # [N, D]
```

- **参数量：** 两者相同。逐基因版没有为每个基因额外建立一张可训练的权重表。
- **权重来源：** 共享评分网络读取不同基因的表示，因此可以给出不同权重；它们通过分类损失反向传播学习。
- **输出大小：** 都是 `[N,D]`，后面的分类器不需要为这项改动更换维度。
- **归一化方向：** node 模式应使用 `dim=1`，使同一基因的九个权重加起来为 1；不能使用 `dim=0` 让不同基因互相竞争。
- **读取权重：** global 的 `beta[0]` 是第一条元路径的权重；node 的 `beta[0]` 是第一个基因的整组元路径权重。
- **效果边界：** 不同基因的权重允许不同，也可能学成相近值；“能个性化”并不自动等于预测更好。

### 2.5 我应该先跑哪两组

> **只验证这一项改动：先比较 E1 和 E2。** 两组都用 GATv2、同一数据和划分、同一随机种子，并关闭中间交互模块。E1 用 global，E2 用 node。

```bash
python run_ablation.py --experiments E1 E2 --seeds 42 --devices 0 --output results/global_vs_node
python summarize_ablations.py results/global_vs_node --output results/global_vs_node_summary
```

以上命令在服务器运行；多卡时将 `--devices 0` 改为相应设备列表。完成后按第 6 节查看不同基因的权重。比较预测 AP 和多个种子的稳定性后，再决定是否加入 E3/E4 的跨视图交互。

### 2.6 与中间融合模块的关系

`fusion.py::SemanticAttention` 已实现开关。`model.py::MyLayer` 在图编码后调用 `ViewInteraction`，再调用语义聚合。

```python
model = MCDHGN(
    num_meta_paths=9, in_size=features.shape[1],
    hidden_size=256, out_size=2, num_heads=[4], dropout=0.4,
    use_gatv2=True,
    semantic_mode="node",       # global / node
    fusion_mode="cross",        # none / cross / pointwise
    fusion_alpha=0.6,            # 0.4 * 编码表示 + 0.6 * 交互表示
    fusion_dim=64,
)
```

默认 `semantic_mode="global", fusion_mode="none"` 保持旧模型行为与旧参数键兼容。默认返回的 beta 仍为 `[M,1]`；node 模式返回 `[N,M,1]`。不要再把 `beta[0]` 当作所有基因的权重。

跨视图模块按每个基因单独计算：

```text
Q = Z Wq, K = Z Wk, V = Z Wv                  [N,M,r]
A = softmax(Q K^T / sqrt(r), dim=-1)           [N,M,M]
shared = (A V) Wo                             [N,M,D]
Z' = (1-alpha) Z + alpha shared
```

注意力只在九条元路径之间归一化，不混合不同基因。按基因分块计算，避免构建 N×N 注意力矩阵。

## 3. 八组预设实验

配置集中在 `ablation_config.py`。先不要同时调整学习率、维度或损失。

1. **E0：GAT + 全局权重 + 无交互。** 在新评估协议下重跑原模型。
2. **E1：GATv2 + 全局权重 + 无交互。** 新评估协议下的 GATv2 基线。
3. **E2：GATv2 + 逐基因权重 + 无交互。** 与 E1 比较，隔离逐基因权重的贡献，参数量完全相同。
4. **E3：GATv2 + 全局权重 + 跨视图交互，alpha=0.6。** 与 E1 比较，隔离交互模块的贡献。
5. **E4：GATv2 + 逐基因权重 + 跨视图交互，alpha=0.6。** E1–E4 构成完整 2×2 因子设计。E4−E3 验证交互后的逐基因权重；E4−E2 验证逐基因权重下的交互收益。
6. **E5：E4 去掉保留分支，alpha=1。** 与 E4 参数量相同，比较是否值得保留图编码表示。
7. **E6：逐基因权重 + 每个视图独立的 MLP，alpha=0.6。** MLP 为 `D→2r→D`，没有跨视图信息传递。它与 E4 交互模块都恰好有 `4Dr` 个参数，缓解“只是增加参数和非线性”的混淆。计算量和函数形式并非完全相同。
8. **E7：GAT + E4 的融合模块。** E7−E0 判断融合对 GAT 是否有效；E4−E7 判断加入融合后 GATv2 是否仍有额外价值。

还可计算交互项 `(E4−E3)−(E2−E1)`：大于零支持两种改动存在正向配合，但有限样本下不能仅凭符号宣称显著。

E1–E6 使用相同种子时，共有编码器、评分 MLP 和分类器的初始参数保持一致；可选融合初始化不会消耗后续共有模块的随机数。CUDA scatter 等操作仍可能造成非确定性。

## 4. 运行步骤

以下命令均从**服务器项目根目录**执行。先激活原来的可用训练环境；不需要同步 `.venv-ablation`、`.cache-ablation` 或 `.python-ablation`。

### 第一步：小图检查，不使用真实数据

```bash
python run_ablation.py --list
python -m pytest -q tests/test_ablation.py tests/test_summary.py
python run_ablation.py --smoke --cpu --experiments E0 E1 E2 E3 E4 E5 E6 E7 --output results/smoke
python summarize_ablations.py results/smoke --expected-folds 2 --output results/smoke_summary
```

`--smoke` 固定为 24 个节点、3 个视图、2 折、2 epoch，检查保存/读取 checkpoint、梯度、指标和权重导出。它不能用来比较准确率。若无 pytest，可先运行 smoke；数值一致性测试仍建议在正式训练前补做。

测试覆盖全局模式与旧公式一致、逐基因权重归一化、不串基因、分块前向/梯度一致、共有参数初始化、MLP 参数量对照、八个模型的反向传播，以及省显存 GATv2 与 DGL 标准实现的前向和梯度对齐。

### 第二步：跑核心实验，先用一个种子

```bash
python run_ablation.py --experiments E0 E1 E2 E3 E4 --seeds 42 --devices 0 1 2 --output results/core
python summarize_ablations.py results/core --output results/core_summary
```

单卡可用 `--devices 0`；不写 `--devices` 则自动检测 CUDA；无 CUDA 时走 CPU，正式图不建议在 CPU 上长时间训练。多卡沿用当前按元路径分配的方式，第一张指定 GPU 负责融合和分类。

所有组默认 hidden=256、heads=4、dropout=0.4、Adam lr=0.01、weight_decay=1e-4、正类权重=1.904、最多1000 epoch，每10 epoch验证一次，连续20次验证AP没有改善则早停。设置 `--patience 0` 可关闭早停，但必须所有对照组一致。

如果预算很紧，可以先对所有组共同使用 `--hidden-size 128` 并放入不同输出目录筛选；必须对最终候选和基线一起恢复 256 维重跑，不能将不同维度直接归因为融合收益。不要只看上次收益最大的 fold 0 来挑方案。

### 第三步：机制对照与种子重复

若 E4 在核心实验中有价值，再运行 E5–E7。然后对需要保留的对照使用相同的 3–5 个种子复核。例如完整三种子方案：

```bash
python run_ablation.py --experiments E5 E6 E7 --seeds 42 --devices 0 1 2 --output results/core
python run_ablation.py --experiments E0 E1 E2 E3 E4 E5 E6 E7 --seeds 43 44 --devices 0 1 2 --output results/core
python summarize_ablations.py results/core --output results/core_summary
```

按你上一轮记录，GATv2每次五折约5.4小时；三种子、八组的总成本可能达到数天，早停后的成本以新日志为准。建议分阶段运行，不要一次启动全部组争抢显存。

完整运行过的同配置种子/折会自动跳过。配置、数据或训练代码哈希不一致会拒绝覆盖；被中断且没有 `result.json` 的目录也会拒绝覆盖，**不支持中途 epoch 续训**。请使用新的 `--output` 目录，只运行未完成的 `--experiments / --seeds / --folds`。汇总支持多个根目录，但不能重复收录相同实验的同一 seed/fold。

```bash
python run_ablation.py --experiments E4 --seeds 43 --folds 3 4 --devices 0 1 2 --output results/retry
python summarize_ablations.py results/core results/retry --output results/combined_summary
```

运行中的粗略查看可加 `--allow-partial`；正式判断必须使用完整、配对一致的折和种子。

### 第四步：预先选定方案后再看测试集

训练阶段不读取测试标签。新脚本按验证 AP 选择唯一 checkpoint，再从该 checkpoint 导出所有指标。AP 指 `average_precision_score`，不会混用梯形 PR-AUC；正类得分统一使用两类 softmax。

正式选定候选后，预先固定要评估的种子和折，对基线与候选使用相同集合。例如预先指定 seed42/fold0，不能从五个测试结果里挑最好的：

```bash
python run_ablation.py --test-run results/core/E1/seed42/fold0 --devices 0 1 2
python run_ablation.py --test-run results/core/E4/seed42/fold0 --devices 0 1 2
```

更完整的报告应在选定模型后评估所有预先指定的种子/折，保留每组结果，不能据测试结果继续选择结构。这里评估的是验证阶段保存的模型；脚本没有自动把全部开发集重训或进行五模型集成。若之后增加全开发集重训，应先固定训练轮数和协议，再评估测试集。

这仍是全图传导式设置：允许使用未标记基因的特征和图结构，训练损失只使用训练标签。不能把结果表述为对未见网络或新基因的归纳泛化。

## 5. 输出在哪里、怎样判断值得借鉴

每个 `results/core/E*/seed*/fold*/` 下会产生：

- `config.json`：配置、代码哈希、图/采样/划分文件 SHA256。
- `history.jsonl`：每次验证的 AP、AUC、ACC、Macro-F1、损失。
- `best.pt`：验证 AP 最优模型。
- `result.json`：同一最佳模型的指标、最佳 epoch、参数量、耗时、各 GPU 峰值已分配显存、环境版本。
- `validation_predictions.npz`：基因ID、标签、softmax正类分数和两个logit。
- `gene_metapath_weights.npz`：`node_ids`、`beta[N,9]`、元路径代码顺序。
- 显式测试后才生成 `test_result.json` 和 `test_predictions.npz`。

原始缓存的 DGL block 会根据保存的源/目标 NID 恢复全局基因编号，防止局部编号与特征行错配；保留全部原始边。不能与旧脚本历史最优日志直接混合比较，E0/E1必须按新协议重跑。

`summarize_ablations.py` 生成逐运行 CSV 和完整 JSON，输出指定对照的 AP/AUC 差值、赢了多少折、每个种子的平均差值，并计算2×2交互项。标准差使用样本标准差 `ddof=1`，单样本标记为 null，不伪造0。先在每个种子内汇总五折，再观察跨种子的稳定性。折间训练集重叠，不把15个fold当作15个独立样本做普通t检验。

建议在运行前约定实际有意义的收益门槛，例如 **AP绝对增益至少0.003（0.3个百分点）**，这是预算取舍示例，不是统计显著性的通用标准。结合以下条件判断：

1. E2−E1 在多个种子中稳定为正且成本接近：支持借鉴逐基因权重。
2. E4−E2 为正，且 E4−E6 也为正：支持跨视图信息交互，而不仅仅是增加相同数量的参数。
3. E4−E5 为正：支持保留图编码表示。若没有改善，不强行保留alpha=0.6。
4. E4−E7 很小、E7耗时明显更低：可能优先保留 GAT + 改进融合，而不坚持 GATv2。
5. 预先选定候选在独立测试标签上的表现也支持上述趋势；同时检查Macro-F1、Top-K候选精度、运行时间及显存。不能只追求某一折的最好数字。

如果均值增益主要由单折驱动，应增加重复实验并检查该折标签/图度分布。注意力权重有差异只能证明模块学到了不同的分配，不能直接证明生物因果解释。

## 6. 查看某个基因的权重

```python
import numpy as np

data = np.load("results/core/E4/seed42/fold0/gene_metapath_weights.npz")
gene_id = 100  # 图中的 Gene 节点ID；基因符号需使用原始项目的ID映射
row = np.flatnonzero(data["node_ids"] == gene_id)[0]
ranking = sorted(zip(data["metapaths"], data["beta"][row]), key=lambda item: -item[1])
print(ranking)
print("sum:", data["beta"][row].sum())
```

全局模式导出的是广播后的相同权重；逐基因模式应检查 `std_by_path` 是否接近零、权重是否极端集中，以及不同种子下同一基因的高权重元路径是否稳定。权重塌缩时，再单独设计温度或全局/局部混合实验，不把它们混进首轮对照。

## 7. 新增数据与结构实验如何分开

新增数据的准备清单、字段、官方下载入口和接入步骤见 [DATA_EXPANSION.md](DATA_EXPANSION.md)。先固定一种架构验证数据收益，再测试“新数据 × 逐基因权重”的组合，避免同时改变数据和模型后无法解释提升来自哪里。

该文档中的 D0–D9 是新增数据实验设计编号，**不是** `run_ablation.py --experiments` 当前支持的参数。现有命令仍使用 E0–E7；同一架构在不同数据目录运行时，通过 `--data-root` 和独立 `--output` 区分。数据变化后应分别汇总；现有汇总脚本会拒绝混合不同数据哈希。
