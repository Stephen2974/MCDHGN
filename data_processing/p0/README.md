# P0 数据处理：执行结果与复跑说明

以本地 `D:\MCDHGN` 数据为准。时间为 Asia/Shanghai。逐步操作和失败记录见 `data_processing/p0/EXECUTION_LOG.md`；机器日志为 `data_versions/p0_v1/events.jsonl`。

## 目录与入口

所有命令从项目根目录运行，使用 `python -m data_processing.p0.<模块名>`，不直接运行子目录中的 `.py` 文件。

```text
data_processing/
  __init__.py
  p0/
    __init__.py
    common.py          # 项目路径、日志、文件与哈希工具
    prepare.py         # 冻结输入、下载 HGNC 快照
    audit.py           # 身份、特征、标签、划分审计
    validate_graph.py  # 实际 DGL 图与缓存核验
    release.py         # 封存及完整性验证
    README.md          # 本说明
    EXECUTION_LOG.md   # 持续追加的执行日志
```

数据产物保留在 `data_versions/p0_v1/`；测试保留在 `tests/test_p0.py`。

## 当前交付边界

本地数据审计及实际 DGL 图核验已执行。原始图、节点 ID、特征、标签和划分均未覆盖。

用户已确认历史负例原始输入没有保留；保留旧标签/划分用于复现，严格版只生成候选清单，不发布新的训练/验证/测试缓存。候选文件中的 `training_eligible=false` / `decision_status=proposed_not_released` 不可擅自改成已批准训练标签。

服务器尚未连接，没有启动服务器训练，也没有运行最终测试评估。旧测试集使用历史未知，不能宣称独立最终测试。

## 已验证事实

- 原始特征表 13,627 个基因；现有节点 12,944 个；原图特征为 12,944 × 48。
- CSV 转 float32、特征字典按 node_id 排列、图内 Gene.feature 三者逐元素完全一致，最大差为 0。没有发现已有特征错位。
- HGNC 有 12,939 个唯一解析节点；U2AF1L5、CBSL 未匹配；TAZ 存在两个候选身份；MYO18A 与 TIAF1 映射到相同 HGNC 身份。保留所有旧节点，不合并、不猜选；这些节点不能自动接收新映射数据。
- 九份缓存的原始节点 ID 完整，转换到全局 ID 不丢边；每份有 12,944 条自环，无零入度节点；所有非自环采样边均可通过其指定原始元路径到达。不能据此追溯历史采样种子或证明原始数据库边的生物真实性。
- 旧正例 2,226，负例代理 4,240，二者无交集。开发集 5,820、测试集 646；五个验证折各 1,164，恰好覆盖开发集一次，开发/测试不重叠。已解析 HGNC 身份未发现跨集合泄漏；未解析身份仍是限制。
- 550 个严格阳性候选要求同时有 NCG canonical 名单及带数字 PMID 的 canonical 记录：542 个来自旧正例，8 个原本未标注。该筛选不等于已逐篇验证论文，也不等于已确定首次驱动证据时间。
- 382 个旧负例与所审查癌症相关来源有交集。这是待核查证据，不是 382 个已确认误标。CancerMine 文本挖掘、目录成员关系与 canonical 证据分别记录。
- 负例代理候选仅是排查清单；原始负例构造不完整，不能据此声称严格真阴性。OncoKB 只作来源盘点，不决定新训练成员。

## 产物导航

`data_versions/p0_v1/` 中：

- `baseline_manifest.json`、`baseline_files.tsv`：改代码前冻结的 92 个本地文件指纹，包含原始数据与当时实际代码。指纹不是二进制备份，需保存整个数据目录。
- `sources/hgnc/`：官方 HGNC 快照和下载元数据。下载日期 2026-10-23；HTTP Last-Modified 为 2026-10-16，GCS generation 为 1792157775531227，SHA256 已固定。此下载时间不是基因证据日期。
- `gene_map.tsv`、`mapping_report.json`、`mapping_conflicts.tsv`、`unmapped.tsv`：稳定身份及异常清单。
- `feature_schema.tsv`、`feature_alignment_report.json`：48 列含义与逐元素对齐结果；历史特征单位/来源版本未知项保持未知。
- `labels_provenance.tsv`：每条来源记录一行，包含原文件及行号、校验值、PMID、癌种、证据类别和使用状态。
- `labels_summary.tsv`、`label_conflicts.tsv`、`label_mapping_issues.tsv`：按节点汇总、冲突线索及映射异常。癌基因/抑癌双重角色可能是上下文差异，不自动视为错误。
- `candidates/`：严格阳性和未发布的负例代理候选，均不能直接拿来训练。
- `splits/`、`split_manifest.json`：旧划分的可读版本；未生成新划分。
- `graph_validation_local.json`：本地真实 DGL 图及九份缓存核验。
- `sources_review.md`、`source_registry.json`：来源版本、字段及使用条件核查；未知信息显式保留。
- `temporal_validation_protocol.json`：尚未建立时间外验证的原因及以后准入规则。
- `quality_report.json`：原始审计汇总；`manifest.json`：整理前的封存指纹，原样保留；`layout_manifest.json`：目录整理后的代码与产物清单，引用原封存清单的 SHA256，验证命令自动优先使用。
- `graph_validation_layout.json`：整理后的模块入口对真实图及缓存的复核；旧报告中的根目录脚本名为历史记录。

日志为追加文件，不纳入封存哈希。未来服务端报告使用新文件名，不覆盖本地核验报告。

## 代码改动

1. 数据处理代码统一放在 `data_processing/p0/`：`prepare.py`、`audit.py`、`validate_graph.py`、`release.py`、`common.py`；通过 Python 模块入口运行。
2. `preprocess/generate_network.ipynb` 用现有 node_id 显式排列特征，验证映射、特征键集合、行数和有限值；没有重新运行会覆盖旧图的单元格。
3. `run_ablation.py` 将 `gene_toids.pkl` 纳入训练输入指纹，验证节点 ID 连续唯一及特征行数；数据版本根目录若有 `gene_map.tsv`、`feature_schema.tsv`、`metapaths.json`、`split_manifest.json`，也纳入指纹。
4. 新增针对别名歧义、重复身份、特征重排、划分遗漏、跨集合身份泄漏和错误标签的测试。

训练加载器的指纹变化会拒绝混用旧 checkpoint/旧输出；新基线必须使用独立输出目录。已有用户代码改动保留，没有提交 Git 或重置工作区。

## 本地核查

当前可用环境：`.venv-ablation`，Python 3.11.9、PyTorch 2.2.2 CPU、DGL 1.1.2。不要为读取旧缓存擅自升级 DGL。

```powershell
$env:DGLBACKEND='pytorch'
& .\.venv-ablation\Scripts\python.exe -m data_processing.p0.release verify
& .\.venv-ablation\Scripts\python.exe -m pytest tests -q
```

需要重新生成审计时应使用新的版本目录；不要覆盖已封存的 `p0_v1` 产物。执行顺序为 `python -m data_processing.p0.prepare freeze` → `python -m data_processing.p0.prepare download-hgnc` → `python -m data_processing.p0.validate_graph` → `python -m data_processing.p0.audit` → 补充来源/协议记录 → `python -m data_processing.p0.release seal`。旧版冻结清单不要重新生成。

## 服务器操作

传输整个项目及数据、`Intermediate/`、`data_versions/p0_v1/` 和当前代码；保留相对路径和原始字节，避免 Git 换行转换导致字节指纹变化。Windows 虚拟环境不要复制为 Linux 运行环境。以下操作由服务器已有兼容环境执行：

```bash
export DGLBACKEND=pytorch
python -m data_processing.p0.release verify
python -m pytest tests -q
python -m data_processing.p0.validate_graph --report data_versions/p0_v1/graph_validation_server.json
```

三步都通过才运行旧协议 D0 的开发集五折复现（不读取测试标签进行模型选择）：

```bash
python run_ablation.py --experiments E1 --seeds 42 --data-root . --devices 0 --output results/p0_legacy_D0
python summarize_ablations.py results/p0_legacy_D0 --output results/p0_legacy_D0_summary
```

保持旧标签、划分、48 维特征和九路径。现有默认正类权重 1.904 继续保留，只用于旧协议对照；未来更改标签后必须重新明确权重策略。`p0_legacy_D0` 表示有明确来源限制的旧协议基线，不是严格标签基线。先在验证集选择 checkpoint；测试集使用历史未核实前不执行 `--test-run` 并宣称独立验证。

本轮未新增 `D0` 命令参数，D0 是数据方案名称；实际模型参数仍为 `E1`。

## 后续必须解决

历史负例原始文件由用户确认无法追回。要发布严格版，需要另行定义可复现的新负例代理策略及适用边界，不能声称恢复了原规则。NCG 来源条件、其他来源版本与原论文身份仍需补齐。时间外验证还需训练证据截止日、逐条首次证据核验和输入知识库时间核查；没有合格数据时保持未建立状态。
