# D1 CNV 数据准备

状态：**数据准备完成，不可直接训练**。生成时间 2026-11-03T16:45:12+08:00。

本目录是 TCGA PanCancer Atlas 2018 离散 GISTIC 频率的审计结果。缺失值保持空白，没有插补，没有生成 `data_variants/cnv_v1`，没有训练。

## 来源

- 仓库：https://github.com/cBioPortal/datahub
- 使用说明：固定提交的 DataHub README 说明这是供 cBioPortal 直接载入的公开暂存数据，并用 Git LFS 保存大文件；仓库提供逐研究 zip 或 git-lfs 下载方式。本任务按固定提交逐文件核对，不使用 zip，也不凭单个 PUBLIC 标记替代来源记录。
- 固定提交：`860b63a7f2d23540c94e25dd35302e133c29f48b`。这是 DataHub 文件快照，不是 2018 队列的生物学日期。
- 研究批次：TCGA PanCancer Atlas 2018（研究 ID 后缀 `tcga_pan_can_atlas_2018`，meta 中 citation 为 TCGA, Cell 2018）。
- 每个研究只使用 `meta_cna.txt` 确认为 `DISCRETE` / `gistic` 的 `data_cna.txt`。连续拷贝数文件即使存在也未下载、未混入。
- LFS 地址来自该提交的 `.lfsconfig`。签名下载链接没有写入清单或日志。
- 基因身份使用 P0 已固定的 HGNC 快照和 `gene_map.tsv`，没有重新下载 HGNC。

## 规则

- 只保留临床字段与 TCGA barcode 都表明实体原发灶（编码 01）且矩阵中确有 CNA 的样本。矛盾样本排除并写入原因。
- 同一患者多个合格样本在计算事件前按 `sample_id` 字典序保留一个。
- COAD/READ 来自 `coadread_tcga_pan_can_atlas_2018`。ONCOTREE 为 COAD 或 READ 时按其分类；MACR 只在部位明确为 Colon 或 Rectum 时分配；字段矛盾或无法判断则排除。同一患者不会同时进入 COAD 和 READ。
- 分母是选中患者中该基因 GISTIC 属于 -2、-1、0、1、2 的人数。`NA` 不进入分母，观测到的 0 进入分母。分母为 0 时频率留空，不写成 0。
- Entrez 经 HGNC 映射到唯一 `resolved` 节点。符号只在无歧义时回退。冲突重复行被隔离，不取最大值或任意一行。

## 本次检查器修复

- 表级验收现在先核对每张表的行数、列契约、node_id 顺序和身份字段，再逐单元格检查掩码、计数、频率、缺失原因和选中患者上限。
- `python -m data_processing.d1.pipeline verify` 是只读离线验证：核对固定提交、120 个来源文件、LFS OID、登记输出、处理代码和 P0 保护文件，不会更新清单。
- 旧 manifest 已保存在 `audit/manifest_before_validator_fix.json`；`audit/validator_fix_relation.json` 记录本次只修复检查器、沿用原数据并由修复后的检查器重新验证。
- 训练图脚本现在要求可追溯的缺失处理协议、冻结映射和 80 列 schema，并在临时目录完成验证后才发布；当前缺失未定，不会生成正式训练图。

## 命令

在项目根目录执行：

```powershell
python -m data_processing.d1.pipeline verify
python -m unittest tests.test_d1
python -m data_processing.d1.build_graph --features data_versions/d1_cnv_v1/cnv_features.tsv
```

`build_graph` 在缺失未处理或没有显式缺失协议时会拒绝写训练图。

## 癌种覆盖

- KIRC（kirc_tcga_pan_can_atlas_2018；OncoTree 类型 ccrcc；Kidney Renal Clear Cell Carcinoma (TCGA, PanCancer Atlas)）：选中患者 509；该癌种有观测的节点 12889 / 12944。 临床文件 OncoTree 计数：CCRCC 512。
- BRCA（brca_tcga_pan_can_atlas_2018；OncoTree 类型 brca；Breast Invasive Carcinoma (TCGA, PanCancer Atlas)）：选中患者 1070；该癌种有观测的节点 12882 / 12944。 临床文件 OncoTree 计数：IDC 780, ILC 201, BRCNOS 77, IMMC 17, MBC 8, BRCA 1。
- READ（coadread_tcga_pan_can_atlas_2018；OncoTree 类型 coadread；Colorectal Adenocarcinoma (TCGA, PanCancer Atlas)）：选中患者 155；该癌种有观测的节点 12886 / 12944。 临床文件 OncoTree 计数：COAD 378, READ 155, MACR 61。
- COAD（coadread_tcga_pan_can_atlas_2018；OncoTree 类型 coadread；Colorectal Adenocarcinoma (TCGA, PanCancer Atlas)）：选中患者 437；该癌种有观测的节点 12886 / 12944。 临床文件 OncoTree 计数：COAD 378, READ 155, MACR 61。
- PRAD（prad_tcga_pan_can_atlas_2018；OncoTree 类型 prad；Prostate Adenocarcinoma (TCGA, PanCancer Atlas)）：选中患者 489；该癌种有观测的节点 12880 / 12944。 临床文件 OncoTree 计数：PRAD 494。
- STAD（stad_tcga_pan_can_atlas_2018；OncoTree 类型 stad；Stomach Adenocarcinoma (TCGA, PanCancer Atlas)）：选中患者 438；该癌种有观测的节点 12883 / 12944。 临床文件 OncoTree 计数：STAD 165, ISTAD 81, TSTAD 79, DSTAD 72, MSTAD 22, SSRCC 13。
- HNSC（hnsc_tcga_pan_can_atlas_2018；OncoTree 类型 hnsc；Head and Neck Squamous Cell Carcinoma (TCGA, PanCancer Atlas)）：选中患者 517；该癌种有观测的节点 12884 / 12944。 临床文件 OncoTree 计数：HNSC 523。
- LUAD（luad_tcga_pan_can_atlas_2018；OncoTree 类型 luad；Lung Adenocarcinoma (TCGA, PanCancer Atlas)）：选中患者 511；该癌种有观测的节点 12885 / 12944。 临床文件 OncoTree 计数：LUAD 566。
- THCA（thca_tcga_pan_can_atlas_2018；OncoTree 类型 thpa；Thyroid Carcinoma (TCGA, PanCancer Atlas)）：选中患者 497；该癌种有观测的节点 12891 / 12944。 临床文件 OncoTree 计数：THPA 500。
- BLCA（blca_tcga_pan_can_atlas_2018；OncoTree 类型 blca；Bladder Urothelial Carcinoma (TCGA, PanCancer Atlas)）：选中患者 408；该癌种有观测的节点 12884 / 12944。 临床文件 OncoTree 计数：BLCA 411。
- ESCA（esca_tcga_pan_can_atlas_2018；OncoTree 类型 esca；Esophageal Adenocarcinoma (TCGA, PanCancer Atlas)）：选中患者 182；该癌种有观测的节点 12887 / 12944。 临床文件 OncoTree 计数：ESCC 95, ESCA 87。
- LIHC（lihc_tcga_pan_can_atlas_2018；OncoTree 类型 hcc；Liver Hepatocellular Carcinoma (TCGA, PanCancer Atlas)）：选中患者 367；该癌种有观测的节点 12885 / 12944。 临床文件 OncoTree 计数：HCC 369, FLC 3。
- UCEC（ucec_tcga_pan_can_atlas_2018；OncoTree 类型 ucec；Uterine Corpus Endometrial Carcinoma (TCGA, PanCancer Atlas)）：选中患者 523；该癌种有观测的节点 12885 / 12944。 临床文件 OncoTree 计数：UEC 399, USC 109, UMEC 21。
- LUSC（lusc_tcga_pan_can_atlas_2018；OncoTree 类型 lusc；Lung Squamous Cell Carcinoma (TCGA, PanCancer Atlas)）：选中患者 487；该癌种有观测的节点 12886 / 12944。 临床文件 OncoTree 计数：LUSC 487。
- CESC（cesc_tcga_pan_can_atlas_2018；OncoTree 类型 cesc；Cervical Squamous Cell Carcinoma (TCGA, PanCancer Atlas)）：选中患者 293；该癌种有观测的节点 12888 / 12944。 临床文件 OncoTree 计数：CESC 251, ECAD 26, CEMU 17, CEEN 3。
- KIRP（kirp_tcga_pan_can_atlas_2018；OncoTree 类型 prcc；Kidney Renal Papillary Cell Carcinoma (TCGA, PanCancer Atlas)）：选中患者 283；该癌种有观测的节点 12893 / 12944。 临床文件 OncoTree 计数：PRCC 283。

## 缺失构成

12,944 × 32 中有 1860 个空白单元格。 来源行没有成功映射：每个癌种有 43 个已解析节点在来源中没有可用的一致基因行； 5 个身份未解析节点（U2AF1L5、MYO18A、TIAF1、CBSL、TAZ）不能从来源接收 CNV。 来源行冲突：另有 162 个基因-癌种位置因重复行测量不一致被隔离，其中 ZFHX3 与 LDB3 在全部癌种被隔离。 患者测量缺失：若选中患者中某基因没有有效 GISTIC，分母可为 0；本轮真实数据中这类单元格为 0 个。 全节点 32 列都有观测的是 12874 个。 没有发现 -2、-1、0、1、2 以外的取值。没有把缺失写成 0。 COAD/READ 临床分配为 oncotree:COAD;site:Colon 375；oncotree:COAD;site:missing 3；oncotree:MACR;site:Colon 61；oncotree:READ;site:Rectum 152；oncotree:READ;site:missing 3；未选中原因：no_cna_data 2。

## 尚未解决

缺失单元格 1860。身份未解析节点 5 个，状态来自 P0 gene_map 的实际 mapping_status。 原 48 维矩阵没有患者 ID，无法核实与本批 CNA 患者是否完全相同。基因组版本未写在 meta_cna.txt 中，因此记录为未知，不从其他文件名推断。 KIRC、KIRP、LIHC、THCA 的 meta type_of_cancer 分别是 ccrcc、prcc、hcc、thpa；研究 ID 仍是 TCGA 项目缩写。 ESCA 研究显示名是 Esophageal Adenocarcinoma，本列仍按该研究中的原发 CNA 样本汇总，没有再按组织学拆开。

## 缺失处理选项（尚未选择）

1. 平台未测到、来源行缺失或患者缺测的基因保持缺失，模型侧明确支持掩码。当前 `run_ablation.py` 拒绝非有限特征。
2. 单独决定“没有成功映射的来源行”是否编码为 0，并保留这份未填充表。这会改变缺失的含义；不能在未记录决定时直接做，也不能把 `no_mapped_cna_row` 直接解释为已经证明平台未测量。
3. 不删除节点，不把 32 维掩码自动拼进主实验变成 112 维。
4. 不做全基因均值、中位数、标准化或 PCA。若以后按折插补，只能在该折训练基因上拟合。
5. 身份未解析的节点不能靠别名自动补上 CNV。

## 以后的同维对照

零值对照和打乱对照都要保留这份 12,944 × 32 的缺失掩码。零值只替换已观测单元格。打乱时在每个特征列内、只在该列已观测的节点之间置换，并记录随机种子；不要把缺失位置一起打乱，避免把覆盖率变化当成 CNV 信号。本轮没有生成或训练这些对照。

## 与 D0 的关系

原 48 维特征来自更早的预处理，文件中没有患者清单。D1 使用上述 2018 研究批次的离散 GISTIC。癌种名称有重叠，但不能据此声称患者队列相同。
