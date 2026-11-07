# D2：突变类型频率（16×3）与位点复发审计

在项目根目录执行。本流程只写入 `data_versions/d2_mutation_v1/`，不改 D0 图、P0/D1 封存文件、标签、划分或训练参数，也不启动训练。

> 工作区说明：历史 P0 文档曾写 `D:\MCDHGN`；当前含 `DATA_EXPANSION.md` / `data_processing` / `data_versions` 的活动工程在 `G:\LLLM\MCDHGN-main`。`D:\MCDHGN` 现为 `author_repo` 摘录，不含本流水线。大文件也落在本仓库的 `data_versions/d2_mutation_v1/sources/`，**禁止**写入 `preprocess_data/gene_mutation/mut_data/`。

```powershell
# 只读核验保护文件（需已有 protected_files_before.json）
python -m data_processing.d2.pipeline verify

# 单元测试（合成数据，与正式产物隔离）
python -m unittest tests.test_d2

# 下载 → 构建 → 封存（顺序执行；download 串行，不占 GPU）
python -m data_processing.d2.pipeline download
python -m data_processing.d2.pipeline build
python -m data_processing.d2.pipeline seal
```

## 数据来源决策

- 本地 `gdc_manifest.mutations.mutect2.2018-11-23.txt` 中的 DR-10.0 Mutect2 UUID 在 GDC Files API 上 **404**，无法复用旧 MAF 字节。
- 改用与 D1 相同的 cBioPortal DataHub 提交 `860b63a7f2d23540c94e25dd35302e133c29f48b` 的 TCGA PanCancer Atlas 2018 `data_mutations.txt`（MUTATION_EXTENDED / MAF）及 `cases_sequenced`。
- 分母 = 各癌种经样本规则选中的去重患者（来自 sequenced case list）。**这是队列级近似，不是逐基因 callable。**
- 频率 **不** 除以基因长度；GENCODE v28 外显子长度只写入 `audit/gene_exon_lengths_gencode_v28.tsv`。
- 沿用 `ultramutated_tumor_ids.txt` 名单排除；旧脚本中的 1000 突变阈值未在旧逻辑中实际使用，此处也不启用。

## 48 维事件类

| 类 | Variant_Classification |
|---|---|
| MISSENSE | Missense_Mutation |
| TRUNC | Nonsense_Mutation, Frame_Shift_Del/Ins, Nonstop_Mutation |
| SPLICE | Splice_Site |

`Splice_Region`、`Silent`、`In_Frame_*`、`Translation_Start_Site` 等记入直方图，**不**并入三类。完整表见版本目录 `classification_map.tsv`。

## 缺失与训练接入

- 映射失败 / 歧义 / 重复身份节点（含 U2AF1L5、CBSL、TAZ、MYO18A、TIAF1）特征留空，`feature_observed=0`。
- 无有效分母时留空，不中位数插补。
- 原始 `d2_mutation_v1` **不宣称可直接训练**（映射受阻节点仍有空单元格）。
- 零填充打包（仅 mapping_blocked，不改已观测值）见：

```powershell
python -m data_processing.d2.build_graph assess
python -m unittest tests.test_d2_build_graph
python -m data_processing.d2.build_graph build
python -m data_processing.d2.build_graph verify-loader
```

产物目录：`data_variants/d2_mutation_zero_v1`（96 维）。`feature_observed` 仍保留原缺失；0 为占位，不是已确认无突变。

## 位点复发

`recurrence/` 下独立产物，**不**拼进 `mutation_type_features.tsv`。候选位点集中度公式与低样本规则见该目录 schema。
