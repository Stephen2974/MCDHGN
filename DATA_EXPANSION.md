# MCDHGN 新数据接入与优化清单

> **建议顺序：先固定基因 ID 与标签证据 → 补 CNV → 拆分现有 C3 → 细分突变类型 / 加 DepMap → 再考虑新网络和蛋白组。**
>
> 这是一份接入设计，不是已获得提升的实验结果。优先级依据本项目的信息缺口、改动范围与验证成本判断。来源页面核对日期：2026-10-15；正式下载时固定 release，不能只记录“最新版”。

本文件可按 Markdown 导入 Notion。每种数据都说明“下载什么、最小字段、处理方式、接入位置、验证对照”。另附交互表格便于按优先级查看；本文件保留全部内容，方便随代码同步到训练服务器。

## 1. 项目现在究竟有哪些数据

### 1.1 基因特征：48 维，不是所有多组学都已经有了

本地 `data/biological_features.csv` 实际是 **Tab 分隔**，应使用 `sep="\t"`。表头为 `Hugosymbol` 加 48 个特征列，共 **13,627 行、13,627 个不同基因符号**，未发现空字符串/NA/NaN 形式的空单元格；这不代表所有值都有完整生物学测量来源。

现有三类输入各 16 列：

- `MF:`：突变频率。
- `METH:`：DNA 甲基化相关特征。
- `GE:`：基因表达相关特征。

按现有表头顺序，16 个癌种是：

```text
KIRC BRCA READ PRAD STAD HNSC LUAD THCA
BLCA ESCA LIHC UCEC COAD LUSC CESC KIRP
```

`preprocess_data/` 的说明和脚本源自 EMOGI 处理流程；其中表达预处理已使用 TCGA 肿瘤与 GTEx 正常数据。因此“再使用 GTEx”未必增加独立信息。当前列中没有 CNV、CRISPR 依赖或蛋白丰度。

### 1.2 节点与特征行必须按 ID 对齐

`data/step1/gene_toids.pkl` 当前包含 **12,944 个基因**，ID 为连续的 0…12943。它少于原始特征表的基因数；`preprocess/gene_protein.ipynb` 会根据网络邻居等条件筛选基因。

第一轮接入保留这套节点映射，只为现有基因补数据。服务器读取训练图后还应验证：

```python
assert graph.num_nodes("Gene") == len(gene_toids)
assert graph.nodes["Gene"].data["feature"].shape[0] == len(gene_toids)
```

这里没有在本机加载 DGL 图或执行训练，因此 12,944 是已核对的**映射条数**，不是另行加载二进制图后测得的节点数。

### 1.3 关系数据：PPI 加八类集合，共九条元路径

`ablation_config.py` 和原训练 notebook 的顺序为：

```text
gg     ：Gene → Gene，CPDB PPI
ga-ag  ：Gene → 位置/细胞遗传带集合 → Gene，C1
gb-bg  ：Gene → 通路集合 → Gene，C2.CP
gc-cg  ：Gene → 调控靶基因集合 → Gene，C3.ALL
gd-dg  ：Gene → 计算得到的基因集合 → Gene，C4
ge-eg  ：Gene → GO 集合 → Gene，C5.GO
gf-fg  ：Gene → HPO 集合 → Gene，C5.HPO
gh-hg  ：Gene → 致癌相关签名集合 → Gene，C6
gi-ig  ：Gene → 细胞类型签名集合 → Gene，C8
```

本地集合版本是 **MSigDB v2023.1.Hs**。除了 Gene，当前主模型并不直接编码这些异类节点的特征：它利用异构图采样出的 Gene–Gene 视图。为非 Gene 节点加向量，不会自动被当前训练流程使用。

> **一个值得先做的整理：** `c_Microrna` 实际接收整个 C3.ALL。本地 JSON 中 MIR 有 2,598 个集合，TFT 有 1,115 个集合。因此现有“miRNA 元路径”已经混入 TF 靶基因集合。下面的 #06 给出拆分实验。

### 1.4 标签和实际训练入口

- 标签源文件位于 `data/labels/`，包括 NCG7.0、OncoKB 及其他来源。
- 当前训练特征读取 `data/network/hetero/new_9nodes_graph.bin`。
- 当前图视图读取 `Intermediate/blocks/0gs.bin` 至 `8gs.bin`。
- 训练/验证划分读取 `Intermediate/label_set/train_batch.pkl`。
- 测试由显式 `--test-run` 读取 `mydatatest_mask.pt` 与 `mydatatest_label.pt`。

**修改 CSV、原始边或正负标签文件，并不自动改变上述缓存。** 需要重新生成与改动相关的产物。

旧 notebook 中还存在 `data/label` 与实际 `data/labels`、文件名带不带 `_0.5` 等路径差异；搬到 Linux 服务器时大小写也区分。先统一到实际文件，不能直接照抄每个旧单元格路径。

原 README 的 CPDB 链接不对应此处的 ConsensusPathDB。其身份和官方地址可由 [ConsensusPathDB 原论文](https://academic.oup.com/nar/article/37/suppl_1/D623/1005985) 确认；本次官方站点直连未成功，不把该旧链接作为可用下载入口。

## 2. 详细数据清单

优先级含义：

- **P0：接入前的基础工作。** 不一定直接提高 AP，但决定结果是否可信。
- **P1：第一轮优先试。** 能补主要缺口，或利用现有数据做有针对性的整理。
- **P2：第二阶段。** 需要更多清洗、缓存重建或额外对照。
- **P3：后续专题。** 先明确癌种/细胞背景，再投入较高成本。

“成本”为相对于本项目的工程工作量估计，不是实际下载大小或训练时长。

### 01. 基因 ID 映射与数据版本表

> **P0 · 基础整理 · 准备成本：低**

**当前情况**：已有 gene_toids.pkl，但缺少统一的跨数据库映射与版本清单。

**需要下载 / 准备**：HGNC complete set（TSV）；保留原项目的 gene_toids.pkl；每个来源的 release、下载日期、文件校验值。

**最小字段**：hgnc_id、symbol、alias_symbol、prev_symbol、ensembl_gene_id、entrez_id、uniprot_ids、status；另存 node_id、原始输入 ID、映射状态和歧义原因。

**处理步骤**：先建立 source_id → HGNC ID → 现有 node_id 的桥接表。Ensembl 去掉版本后缀时保留原 ID；别名一对多不自动选第一项；禁止按表格行号连接。第一轮只给现有节点补数据。

**交给模型的数据**：gene_map.tsv、unmapped.tsv、manifest.json；本项通常不增加特征维度。

**代码接入位置**：data/step1/gene_toids.pkl 是节点顺序基准。generate_network.ipynb 的 gene_feat.values() 应改为显式按 node_id 排列；不重排旧节点。

**可能帮助什么**：减少“数据已经加入但实际贴到错误基因”的问题，使不同版本的覆盖率可比较。

**如何证明值得加入**：先做映射审计。若发现并修复实质错误，建立新基线并重跑全部对照，不能把修复后的结果与旧数据指标直接归因为新模态收益。

**容易出错的地方**：13,627 是特征表行数，12,944 是现有映射条数；实际训练张量行数须在服务器读取图后再断言一致。两个数字不能混用。

**来源**：[HGNC 完整基因映射下载](https://hgnc.genenames.org/download/)；[HGNC 历史版本与字段说明](https://hgnc.genenames.org/download/archive/)。

---

### 02. 标签证据、冲突与独立验证名单

> **P0 · 标签与评估 · 准备成本：中**

**当前情况**：已有 NCG7.0、OncoKB 和其他候选名单；训练实际读取预先保存的划分。

**需要下载 / 准备**：NCG driver annotation/supporting evidence；保存独立发布版本或后续研究的新增确认名单；保留每个标签原始证据。

**最小字段**：gene_id、label_status（positive/negative_proxy/unlabeled）、driver_class（canonical/candidate）、cancer_type、PMID、evidence_date、source_release、role（oncogene/TSG，若有）、conflict_flag。

**处理步骤**：先把 canonical 与 candidate 分开，冲突基因待核查；候选或未收录基因不直接当真负例。冻结训练名单后，再构建不重叠的时间外验证名单。数据库改版日期不能代替首次证据日期。

**交给模型的数据**：labels_provenance.tsv、冻结的 train/val/test 基因名单；时间外新增确认基因的排序评估。

**代码接入位置**：data/labels/ → 重新生成 Intermediate/label_set/train_batch.pkl 和显式测试文件；按 gene_map 转 node_id。新标签实验须新建版本，不能只替换正负 pkl 后沿用旧划分。

**可能帮助什么**：帮助分清模型提升、标签更新和知识泄漏；提高发现新驱动基因这一目标的可信度。

**如何证明值得加入**：保持数据输入不变，比较严格证据标签与原标签的稳定性；时间外集只作最终评估。只有新增阳性时报告 Recall@K/排名富集，不把未知基因当真阴性计算完整分类 AP/AUC。

**容易出错的地方**：现有标签生成 notebook 依赖中间变量，不能仅据文件名断言完整来源。OncoKB 当前 FAQ 明确不允许用于 AI/ML 训练；基准验证需明确许可。已有文件的历史使用条件需单独追溯，本清单不把它列为可直接新增的训练源。NCG 也需记录原始证据来源和使用条件。

**来源**：[NCG 数据下载](https://www.network-cancer-genes.org/download.php)；[NCG 标签与证据字段](https://network-cancer-genes.org/help.php)；[OncoKB 当前数据使用说明](https://faq.oncokb.org/licensing)。

---

### 03. CNV：基因拷贝数扩增与缺失

> **P1 · 新特征 · 准备成本：低至中**

**当前情况**：当前 48 维中没有 CNV；最适合首轮补充的独立分子模态。

**需要下载 / 准备**：首选同一 TCGA 研究版本的 gene × sample 离散 GISTIC CNA 表及样本临床映射；可用 cBioPortal 研究下载包。若用 GDC，选择基因级 Copy Number Estimate 并记录 ASCAT/ABSOLUTE 流程。

**最小字段**：Hugo_Symbol/Entrez_Gene_Id 或 Ensembl gene ID、sample_id、patient_id、cancer_type、sample_type、CNA value、workflow、genome_build；有连续拷贝数时另存 purity/ploidy。

**处理步骤**：第一轮按原 16 癌种统计每个基因高扩增频率 P(GISTIC=+2) 和深度缺失频率 P(GISTIC=-2)。分母为该基因实际有有效测量的去重患者数；同一患者多个样本按预定规则选一个。缺测不当 0。

**交给模型的数据**：cnv_features.tsv：每基因 32 个数值列；例如 CNV_AMP:BRCA、CNV_DEL:BRCA。拼接后为 80 维，缺失掩码如需加入另计。

**代码接入位置**：按原 node_id 对齐后拼接到 Gene.feature，保存一个新的 graph.bin 数据版本；图拓扑和节点顺序未变时复用原 blocks 和 splits。

**可能帮助什么**：检验扩增/缺失证据能否补充原来的突变率、表达和甲基化，尤其适合与逐基因融合共同评估。

**如何证明值得加入**：D1：基线 + 32 维 CNV；另做零值维度对照或打乱基因对应关系的 CNV 对照，保留真实缺失掩码，以排除只增加输入参数或覆盖率偏差。

**容易出错的地方**：GISTIC 的 -2…+2 是类别编码，不是绝对拷贝数；不能把 GDC 的绝对 copy_number==2 当作高扩增。不同平台/流程不直接混合。获得收益是待验证假设。

**来源**：[cBioPortal 数据集入口](https://www.cbioportal.org/datasets)；[cBioPortal 拷贝数与临床文件格式](https://docs.cbioportal.org/file-formats/)；[GDC 拷贝数处理流程](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/CNV_Pipeline/)。

---

### 04. 突变类型与复发模式

> **P1 · 新特征 · 准备成本：中**

**当前情况**：已有每癌种突变频率，但尚未显式区分功能类型。

**需要下载 / 准备**：GDC 开放的过滤体细胞 MAF、对应病例和样本清单、与原流程一致的基因长度注释；先检查现有 MAF 是否可复用。

**最小字段**：Hugo_Symbol、Entrez_Gene_Id、Tumor_Sample_Barcode、Variant_Classification、Variant_Type、HGVSp_Short（若提供）、染色体/坐标/等位基因、FILTER、参考基因组、sample/project ID。

**处理步骤**：预定义错义、截断、剪接等类别，统计发生至少一次相应改变的患者比例；多转录本/重复位点去重。将蛋白位点复发度作为后续独立实验，保留基因长度和可测覆盖信息。

**交给模型的数据**：例如 16 癌种 × 3 类 = 48 维新特征；原 48 维基础上变成 96 维。新增总维度以实际列清单为准。

**代码接入位置**：扩展 preprocess_data/gene_mutation 的输出，再与原 node_id 对齐；仅增加特征时不改元路径。

**可能帮助什么**：同样的总突变率可能来自不同突变模式，类型细分可能提供额外判别信息。

**如何证明值得加入**：D2：仅加类型统计，与 D0 比较；确认有用后再测 D1+D2。另比较“替换原 MF”与“保留 MF 并拼接”，检验冗余。

**容易出错的地方**：MAF 没记录不等于该基因被充分测序且无突变；病例分母应来自可检测样本清单。不能把已知 driver 注释/数据库标签直接混入突变输入。

**来源**：[GDC MAF 字段说明](https://docs.gdc.cancer.gov/Data/File_Formats/MAF_Format/)；[GDC 数据门户与下载说明](https://gdc.cancer.gov/access-data/gdc-data-portal)。

---

### 05. DepMap：癌细胞系 CRISPR 依赖

> **P1 · 新特征 · 准备成本：中**

**当前情况**：当前没有实验扰动后的基因依赖测量。

**需要下载 / 准备**：同一发布批次的 CRISPRGeneEffect.csv、Model.csv，以及该版 README/质量字段；可选基因依赖概率表。门户可能需要浏览器验证。

**最小字段**：ModelID、基因标识（常含符号和 Entrez）、Chronos gene effect、细胞系谱系/疾病亚型、模型类型（2D/3D 等）、质量标记；保留实际有筛选数据的模型列表。

**处理步骤**：按模型 ID 联结，先限定可比较的模型类型；每个谱系计算基因依赖均值与标准差。建议预定最少模型数（如 5，仅是实验阈值），不足记缺失。先按谱系分别标准化，勿把不同 release 混合。

**交给模型的数据**：若有 L 个足够覆盖的谱系，得到 2L 个数值列，另附覆盖计数/缺失表；不强求覆盖全部 16 癌种。

**代码接入位置**：低成本路线是拼接 Gene.feature；先不同时新增共依赖图。保留 gene × lineage 的语义映射。

**可能帮助什么**：提供功能扰动证据，与静态组学、知识图谱互补的可能性较高。

**如何证明值得加入**：D3：基线 + 谱系依赖；报告全部基因和预先定义的 common-essential 分层结果，判断是否只学到普遍生存必需性。

**容易出错的地方**：细胞系依赖不等于肿瘤起始驱动；抑癌基因不一定在敲除筛选中表现为依赖。Model.csv 有模型但不一定有 CRISPR 测量；版本更新可能改变校正与模型组成。

**来源**：[DepMap 数据下载入口](https://depmap.org/portal/data_page/?tab=allData)；[DepMap 团队对 GeneEffect 与 Model 的说明](https://forum.depmap.org/t/discrepancies-between-downloaded-files-and-gui/3269)；[DepMap 作者对模型与 Chronos 数据的说明](https://www.nature.com/articles/s41586-026-10843-7)。

---

### 06. 拆分现有 C3：miRNA 与 TF 靶基因集

> **P1 · 已有数据整理 · 准备成本：低至中**

**当前情况**：本地 c3.all.v2023.1.Hs.json 已含 2,598 个 MIR 和 1,115 个 TFT 基因集；它们被一起装入 c_Microrna。

**需要下载 / 准备**：不需要先下载新库：读取现有 JSON 的 collection 字段，按 C3:MIR:* 与 C3:TFT:* 拆分。

**最小字段**：systematicName/集合名、collection、geneSymbols、exactSource、PMID；这些是靶基因集合，不保证每个集合都对应一个独立实体 miRNA/TF。

**处理步骤**：为 MIR 集合与 TFT 集合建立不同节点类型及 Gene↔set 边；保留同一版本及相同基因全集。拆分时不要同时升级 MSigDB，以便归因。

**交给模型的数据**：原 gc-cg 拆成两条视图；其余不变时 M 从 9 变为 10。

**代码接入位置**：改 preprocess/Msig_preprocess.ipynb 和 generate_network.ipynb；重建相应随机游走 blocks；同步元路径名称与 run_ablation.py 的加载数量。仅改显示名称不会产生拆分。

**可能帮助什么**：元路径语义更清楚，逐基因权重才可能区分两类调控信息。

**如何证明值得加入**：D4：混合集合 vs MIR/TFT 拆分；新增编码器会增加参数，必须再做相同大小的随机二分 C3 对照，或使用共享编码器等容量控制。

**容易出错的地方**：这属于重组已有信息，不能写成新增 TF 数据。不同视图数量的 beta 不直接逐列比较，必须保存元路径名称和版本。

**来源**：[MSigDB 人类集合及下载](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)。

---

### 07. 有证据来源的高质量 PPI

> **P1 · 图关系 · 准备成本：中**

**当前情况**：已有 CPDB PPI，但训练当前只使用缓存中的拓扑。

**需要下载 / 准备**：STRING 人类 physical.links.detailed 与 protein ID 映射，或 BioGRID Homo sapiens TAB 3.0 物理相互作用；两者先独立测试。

**最小字段**：gene/protein A、gene/protein B、species=9606、物理/遗传作用类型、实验方法、来源数据库、PMID；STRING 另保留 combined_score 和各证据通道。

**处理步骤**：限定人类与物理互作，蛋白映射至基因后去重，记录一个基因多蛋白映射规则；先固定节点集替换 gg。阈值只在验证阶段预定，额外做匹配边数或度分布的对照。

**交给模型的数据**：ppi_edges.tsv 与证据表；第一阶段输出筛选后的无权图。

**代码接入位置**：替换 Gene-gg-Gene 边并重建对应采样缓存。现有自定义 GATv2/采样流程没有自动使用 confidence；把 score 存入 edata 并不等于已加权。

**可能帮助什么**：测试减少低可信边是否比单纯增加网络规模更有帮助。

**如何证明值得加入**：D5：原 CPDB vs 新物理 PPI；固定非 PPI 视图和特征。再分开比较高置信过滤、边数匹配，最后才考虑可学习的边权机制。

**容易出错的地方**：STRING functional association 不都等于直接物理互作；不要把 text-mining 通道当独立验证证据。CPDB 已整合多个来源，新库可能大量重叠，应报告新增/交集边数。

**来源**：[STRING 网络与证据通道下载](https://string-db.org/cgi/download)；[BioGRID TAB 3.0 字段说明](https://wiki.thebiogrid.org/doku.php/biogrid_tab_version_3.0)；[ConsensusPathDB 原始论文与官方地址](https://academic.oup.com/nar/article/37/suppl_1/D623/1005985)。

---

### 08. 肿瘤背景下的共表达网络

> **P2 · 图关系 · 准备成本：中至高**

**当前情况**：已有表达差异特征和知识库集合；没有据本次患者表达矩阵构建的独立共表达视图。

**需要下载 / 准备**：GDC 同一流程的 gene × patient RNA 表达矩阵、癌种、样本类型、患者 ID、批次；先挑样本足够的癌种。

**最小字段**：gene_id、sample_id、patient_id、cancer_type、expression_unit、workflow、batch、tumor/normal；可选肿瘤纯度。

**处理步骤**：每癌种分别建相关网络，预先固定低表达过滤和 top-k 邻居数，用患者 bootstrap 检查稳定性；控制批次/纯度影响。先限制为一条汇总视图，避免骤增到 16 个大图。

**交给模型的数据**：coexpression_edges.tsv：src/dst、相关值、符号、样本数、稳定性；正负边分开或第一轮只用稳定正边。

**代码接入位置**：增加 Gene-coexpr-Gene 视图，重建缓存并扩展元路径元数据；当前模型不会自动利用 signed edge 属性。

**可能帮助什么**：有机会让静态 PPI 得到肿瘤状态相关的补充。

**如何证明值得加入**：D6：基线 + 共表达视图，与边数/度分布匹配的重连图比较；新增图编码器的参数开销需记录。

**容易出错的地方**：不能混合所有癌种直接计算相关后把组织差异当调控；相关性不等于调控。若做跨患者或跨队列泛化，图只能用训练队列构建。

**来源**：[GDC 数据门户与下载说明](https://gdc.cancer.gov/access-data/gdc-data-portal)。

---

### 09. 有方向和证据等级的 TF→靶基因调控

> **P2 · 图关系 · 准备成本：中**

**当前情况**：已有 C3 TF 靶基因集合，但不是同一种有方向、有符号的调控边。

**需要下载 / 准备**：DoRothEA/OmniPath 人类 TF-target 数据及证据等级；首轮可固定 A/B 等级。

**最小字段**：TF ID、target ID、confidence、mode of regulation（激活/抑制，如有）、sources、references、organism。

**处理步骤**：先映射到现有节点，保留方向；可将入边与出边作为不同视图，激活/抑制另存。记录哪些边和 C3 共靶关系实际重复。

**交给模型的数据**：regulatory_edges.tsv；一条 TF→target 直接图，或预先定义的入/出视图。

**代码接入位置**：新增 Gene-regulates-Gene 关系与 blocks；若使用符号和多种边类型，需要显式修改消息传递，不能只保存列。

**可能帮助什么**：检验有方向的调控证据是否补充现有“共享某个集合”的相似关系。

**如何证明值得加入**：D7：先与 D4（C3 已拆分）比较增量，另做打乱边方向的对照。不要把同一批 TF 知识重复计为两项独立贡献。

**容易出错的地方**：A/B 是资源的证据分级，不是逐癌种实测的因果结论；全局调控关系的组织背景可能不匹配。

**来源**：[DoRothEA 调控边与置信等级](https://r.omnipathdb.org/reference/dorothea.html)；[MSigDB 人类集合及下载](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)。

---

### 10. 通路 / GO / Hallmark 的更新与去冗余

> **P2 · 已有数据整理 · 准备成本：中**

**当前情况**：已有 C2.CP、C5.GO/HPO 等，Reactome 和 GO 不是项目缺失的全新类型。

**需要下载 / 准备**：固定 release 的 MSigDB 集合；或 Reactome gene-to-pathway 与层级表、GO 人类 GAF/GPAD 注释；Hallmark 可作为较紧凑的替代视图。

**最小字段**：gene_id、set/pathway/GO ID、namespace、source_release、evidence_code、qualifier、parent-child；保留集合成员与原始出处。

**处理步骤**：先量化现有与新集合的 Jaccard/边重合；限制过大/过小集合的阈值预先固定。GO 处理 NOT、过期 ID 与父子传播，不能把否定注释当正边。

**交给模型的数据**：清理后的 gene-set 边和集合字典；替换现有视图优先于追加重复视图。

**代码接入位置**：更新 Msig_preprocess 与 generate_network 并重建受影响 blocks；若只是原数量视图更新，M 可保持 9。

**可能帮助什么**：检验减少泛化程度过高、重复或过时的连接是否改善有效信息比例。

**如何证明值得加入**：D8：先在原 2023.1 数据上去冗余，再单独测版本更新；对比匹配集合数/边数的随机删减，避免将稀疏化误判为知识更新收益。

**容易出错的地方**：最新 GO/通路可能包含后来发表的癌基因知识；时间外评估必须冻结到截止日期之前。HPO 与疾病相关集合尤其要审计标签来源重叠。

**来源**：[MSigDB 人类集合及下载](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)；[Reactome 通路、层级与基因映射下载](https://reactome.org/download-data)；[GO 注释下载](https://geneontology.org/docs/download-go-annotations/)。

---

### 11. CPTAC 蛋白组与后续磷酸化组

> **P2 · 新特征 · 准备成本：中至高**

**当前情况**：当前无蛋白丰度；RNA 差异不能完全替代蛋白测量。

**需要下载 / 准备**：PDC/CPTAC 处理后的蛋白定量矩阵和 study/sample/biospecimen 元数据，优先有肿瘤与匹配正常组织的研究；不要先下载原始质谱。

**最小字段**：protein group/UniProt/gene ID、sample/aliquot/case ID、cancer_type、tumor/normal、log2 abundance 或 ratio、定量参考、批次、缺失标记；磷酸化另需位点与定位质量。

**处理步骤**：每研究内部预处理，蛋白组到基因的多对多映射保留歧义；有匹配正常时算差异，无匹配正常时不能冒充肿瘤/正常差值。跨研究做可解释汇总而非直接混合强度。

**交给模型的数据**：每癌种的蛋白差异/变异特征与缺失掩码；位点级数据先汇总成基因级统计。

**代码接入位置**：以额外模态拼接 Gene.feature；高维时使用固定规模投影并加等参数对照。

**可能帮助什么**：检验蛋白层面的信号是否补充转录和基因组信息。

**如何证明值得加入**：D9：先做蛋白丰度，再在确有覆盖的癌种测试磷酸化；共同覆盖基因子集和全节点插补结果都报告。

**容易出错的地方**：CPTAC 与 TCGA 不一定是同一批患者；不能按癌种名称伪造样本配对。缺失常不是随机的，盲目零填充容易变成检测率分类器。

**来源**：[PDC 蛋白定量与数据下载指南](https://pdc.cancer.gov/pdc-docs/usage)。

---

### 12. 正常组织表达背景与组织特异性

> **P2 · 新特征 · 准备成本：低至中**

**当前情况**：原表达预处理已使用 GTEx 正常组织；单纯再加同样的 GTEx 差异不是独立模态。

**需要下载 / 准备**：GTEx 组织表达汇总，或 HPA 基因×组织 RNA/蛋白表达；优先补“正常表达水平/组织特异性”，而非重复现有 fold change。

**最小字段**：Ensembl/HGNC ID、tissue、expression、unit、sample_count、release；蛋白 IHC 另保留可靠性与检测方法。

**处理步骤**：映射到对应癌种的正常组织；log1p 后计算每组织表达和特异性统计，记录何时与原 GE 正常来源重叠。

**交给模型的数据**：基因×组织连续特征，或每基因少量特异性指标；与原48维拼接。

**代码接入位置**：只需扩展 Gene.feature；先不把组织特异性又同时转成新图。

**可能帮助什么**：测试正常组织背景能否解释为什么某些基因的同幅度差异具有不同意义。

**如何证明值得加入**：与基线和“仅加入原正常表达水平”对照；如果只因重复 GTEx 信息获益，不宣称发现独立新证据。

**容易出错的地方**：正常 GTEx 与肿瘤 TCGA 有采样/处理差异。HPA 数据也整合外部来源，应按来源拆分，避免把整合后的数据当独立复现。

**来源**：[GTEx 成人组织数据下载入口](https://gtexportal.org/home/downloads/adult-gtex)；[HPA 组织、单细胞和蛋白表达下载](https://www.proteinatlas.org/about/download)。

---

### 13. 细胞类型分辨的表达特征

> **P3 · 新特征 · 准备成本：中至高**

**当前情况**：已有 C8 细胞类型签名集合；缺少按细胞类型定量的表达背景。

**需要下载 / 准备**：先使用 HPA 处理后的 cell-type expression；若研究肿瘤微环境，再选择有患者、恶性状态和细胞注释的公开肿瘤单细胞研究。

**最小字段**：gene_id、cell_type、tissue、表达均值/检测比例、样本数；原始肿瘤研究另需 donor_id、malignant/normal、batch 和注释依据。

**处理步骤**：按患者×细胞类型做 pseudobulk，再跨患者汇总，不能将几万个细胞当作几万个独立患者。先限制到少数相关细胞类型。

**交给模型的数据**：gene × cell-type 特征或特异性统计；不直接把全部单细胞加成异构图节点。

**代码接入位置**：第一阶段只拼接特征；与已有 C8 图同时使用时做互相删除的对照。

**可能帮助什么**：检验 bulk 信号来自肿瘤细胞还是免疫/基质背景；有利于解释元路径权重，但不保证提高泛癌分类。

**如何证明值得加入**：先在选定癌种验证特征质量，再比较保留/移除 C8 时的新增特征收益；固定患者与基因评估集合。

**容易出错的地方**：HPA 的正常细胞图谱不能等同于肿瘤恶性细胞状态；细胞组成与肿瘤驱动之间不能直接作因果推断。

**来源**：[HPA 组织、单细胞和蛋白表达下载](https://www.proteinatlas.org/about/download)。

---

### 14. 独立队列、样本元数据与外部验证

> **P2 · 标签与评估 · 准备成本：中至高**

**当前情况**：当前是全图传导式基因分类；随机基因五折不能单独证明跨队列泛化。

**需要下载 / 准备**：cBioPortal 中明确独立的癌症研究或 GDC 外部项目，收集处理后的组学、临床/样本说明与对应的外部驱动证据；先排查是否仍是同一 TCGA 队列的镜像。

**最小字段**：study/project ID、patient_id、sample_id、cancer_type、assay/panel、gene coverage、处理流程、参考基因组、发布日期、外部标签证据。

**处理步骤**：按患者和研究去重；训练、调参、最终队列分开。只有肿瘤组织的新测量不能自动产生新的基因 driver 真值，必须明确验证终点。

**交给模型的数据**：外部特征版本、队列说明、可比较的覆盖基因集合和独立标签/排序评价协议。

**代码接入位置**：单独数据版本与评估入口；若要求新节点归纳，需另设计训练方式，现有全图 runner 不能直接声称支持。

**可能帮助什么**：衡量收益是否依赖某一队列或知识库，是方法价值验证而非简单多加几个训练特征。

**如何证明值得加入**：选定模型和数据方案后才评估，报告原覆盖集合与新覆盖集合。标签/阳性比例不同的 AP 不能直接当作与原五折等价的提升。

**容易出错的地方**：小基因 panel 未测到的基因不能当零突变或阴性；所谓“不同数据库”可能分发同一患者队列。

**来源**：[cBioPortal 数据集入口](https://www.cbioportal.org/datasets)；[cBioPortal 拷贝数与临床文件格式](https://docs.cbioportal.org/file-formats/)；[GDC 数据门户与下载说明](https://gdc.cancer.gov/access-data/gdc-data-portal)。

## 3. 真正接入时，准备这六类文件

以下名称是**建议新建的统一格式**，不是仓库已经存在的输出，也不是某个数据库保证提供的原始字段名。先在预处理阶段转换。

### 3.1 gene_map.tsv：所有数据共用的基因身份表

```text
node_id  hgnc_id  symbol  ensembl_gene_id  entrez_id  source_id  mapping_status
```

要求一个 node_id 对应一个稳定基因身份。蛋白多对一、旧符号和别名冲突都要留审计记录。第一轮不扩充节点；以后若扩充，需要同时重建映射、特征、图和划分，不能延用旧整数 ID 的标签。

### 3.2 sample_metadata.tsv：样本级数据不可缺的分母

```text
sample_id  patient_id  cohort  cancer_type  sample_type  assay  workflow  genome_build  batch
```

突变、CNV、表达、甲基化、蛋白组等都需要知道测了哪些患者。同一患者多个样本应按预先确定的规则选取；不能只统计“发生事件的患者”，遗漏已检测但没事件的患者。

### 3.3 feature_values.tsv 与 feature_schema.tsv

```text
feature_values.tsv:
node_id  HGNC_symbol  CNV_AMP:BRCA  CNV_DEL:BRCA  ...

feature_schema.tsv:
column_name  modality  cancer_type  unit  statistic  missing_rule  source_release
```

另存 `feature_observed.tsv`，同形状记录是否有有效观测。模型输入不允许残留 NaN/Inf，但插补前要保存缺失事实。

- 首轮 CNV 的频率在 [0,1] 内，可直接先测试，减少额外缩放变量。
- 其他模态先按明确的生物统计定义汇总；如果需要跨基因标准化、PCA 或插补中位数，为严格评估应只在每折训练基因上拟合，并保存变换器给该折验证/测试使用。
- 当前 runner **没有**自动执行这些新预处理。不要先全表拟合 PCA，再把结果称为严格的未见基因评估。
- 高维特征可先比较固定 32/64 维投影；同时记录参数量与相同维度的对照。
- 保留原 48 维作为独立基线，不悄悄改它的归一化或样本集合。

### 3.4 relations.tsv 与 relation_schema.json

```text
src_id  src_type  relation  dst_id  dst_type  score  sign  evidence  source_release
```

在 schema 中声明方向、节点类型、score 的尺度、是否使用权重、是否添加反向边和自环。多个来源支持同一条边时合并来源，避免无意增加重复边的权重。

`model.py::MySampler` 当前随机游走后会去掉旧自环，再给种子节点添加自环。新图保留这一策略或为所有对照统一替换；不能让一种图有自环、另一种没有。

### 3.5 labels_provenance.tsv 与 split_manifest.json

```text
gene_id  label_status  evidence_level  source  source_release  PMID  evidence_date  split
```

不要使用“未被某数据库收录”直接推导“不是驱动基因”。保留负例代理构建规则，独立检查正负冲突。仅增加输入特征时保持原划分；更新标签时建立新协议并重跑基线。

### 3.6 manifest.json 与 quality_report.json

至少记录：

- 来源 URL、release、下载日期、SHA256、原始许可/使用条件说明。
- 文件字段映射、处理代码版本、参数、患者去重规则。
- 输入基因数、成功映射数、歧义/未匹配数、新增维度和缺失率。
- 对现有全部节点、训练阳性、训练阴性的覆盖率；最终测试分层仅在模型锁定后检查。
- 每癌种的有效患者数，每条视图的边数、孤立节点数、度分布和重复边数。
- 与旧数据及其他新来源的患者/边/集合重叠率。

**不预设一个适用于所有模态的覆盖率合格线。** 可先约定例如“保留特征至少覆盖 80% 现有基因”，这是项目筛选规则示例；同时报告覆盖子集表现，避免对稀疏但有价值的数据一刀切。

## 4. 三种接入方式，改动范围不同

### A. 只加特征：CNV / 突变类型 / DepMap / 蛋白组

这是首轮推荐路线。节点顺序和关系保持原样，输出新图文件，其中只有 Gene.feature 改变。现有模型构造使用 `features.shape[1]`，所以输入从 48 变成 80 不需要手写分类器维度；输入投影层参数会相应增加，仍要记录容量变化。

下面是**服务器预处理示例**。假设 `cnv_features_aligned.tsv` 已用 HGNC 映射到旧图使用的基因符号；“已对齐”仍须通过 reindex 检查，而不是相信文件名。本段只演示无缺失的 32 维 CNV，其他模态不能直接照抄零填充。

```python
from pathlib import Path
import pickle
import pandas as pd
import torch
import dgl

source_root = Path(".")
variant_root = Path("data_variants/cnv_v1")

with (source_root / "data/step1/gene_toids.pkl").open("rb") as f:
    gene_toids = pickle.load(f)  # 只读取自己项目中可信的预处理文件
ordered_genes = sorted(gene_toids, key=gene_toids.get)
assert [gene_toids[g] for g in ordered_genes] == list(range(len(ordered_genes)))

extra = pd.read_csv("cnv_features_aligned.tsv", sep="\t", index_col="Hugosymbol")
assert extra.index.is_unique
# 列表中应只有预先定义的32个数值列；患者数/来源等元数据另存。
assert extra.shape[1] == 32
extra = extra.reindex(ordered_genes)
if extra.isna().any().any():
    raise ValueError("先明确缺失原因、插补规则及掩码，不能将缺测直接视作无改变")
x_extra = torch.tensor(extra.to_numpy(), dtype=torch.float32)
assert torch.isfinite(x_extra).all()
assert ((x_extra >= 0) & (x_extra <= 1)).all()

graphs, graph_labels = dgl.load_graphs(
    str(source_root / "data/network/hetero/new_9nodes_graph.bin")
)
g = graphs[0]
x_old = g.nodes["Gene"].data["feature"].cpu()
assert x_old.shape == (len(ordered_genes), 48)
g.nodes["Gene"].data["feature"] = torch.cat([x_old, x_extra], dim=1)

out = variant_root / "data/network/hetero/new_9nodes_graph.bin"
out.parent.mkdir(parents=True, exist_ok=True)
dgl.save_graphs(str(out), [g], graph_labels)
# 还须将原 blocks、label_set 和 gene_toids 按相同相对路径放入 variant_root；
# 保存 feature_schema、gene_map、来源 manifest 后，才是完整数据版本。
```

此示例保留旧数据根目录，不覆盖原图。仅特征变化时原 blocks/splits 可复制或使用服务器支持的链接复用，前提是节点 ID 完全不变。

### B. 换关系，元路径仍为九条：替换 PPI / 更新 GO

除了构建新的异构图，必须**重建受影响的采样图缓存**。模型实际读取的 blocks 改变后，关系更新才生效。

- 固定原基因全集，先只替换一个关系源。
- 采样随机种子、邻居预算保持对照一致；默认 notebook 示例预算为 128。
- 未改变的视图尽量逐文件复用，避免额外随机采样波动。
- 记录变化视图的采样前后边数和自环规则；有必要时重复采样种子检验稳健性。
- 字典、数据文件和 blocks 的 node_id 必须与 Gene.feature 一致。

### C. 增加元路径：拆 C3 / 新增共表达或调控

当前加载器存在 `range(9)`；导出权重时使用 `METAPATHS[:num_paths]`。因此**只多放一个 9gs.bin 不会自动接入第十条视图**。

至少需要同时改：

1. `preprocess/Msig_preprocess.ipynb`：集合整理，保留原集合类别。
2. `preprocess/generate_network.ipynb`：异构关系及节点映射。
3. `model.py::MySampler` 的调用配置：新的元路径列表与对应缓存。
4. `run_ablation.py::load_data`：按显式元路径清单读取全部 blocks。
5. `ablation_config.py::METAPATHS` 与 `run_ablation.py` 权重导出：名称、顺序、数量一致，并加入断言。
6. 数据版本指纹：把元路径清单、gene_map、feature_schema 也纳入清单/哈希。

模型主体已经使用 `num_meta_paths=len(graphs)`，但**当前数据加载和命名仍不是动态的**。推荐后续统一从 `metapaths.json` 读路径和名称，替代多个文件中手工同步数字。本文没有实施这项代码修改。

## 5. 按这个顺序设计实验

这些 **D 编号是数据方案编号，不是现有命令参数**。E 编号仍表示模型结构；先固定 E1（GATv2/global/none）或你已经选定的架构，所有数据方案共用该架构。

### 第一轮：一项一项加入

- **D0：** 原 48 维 + 原九条视图 + 冻结标签。按修正后的评估流程重跑基线。
- **D1：** D0 + CNV 的 32 维频率；另做零值/打乱对应的同维对照。
- **D2：** D0 + 突变类别统计。
- **D3：** D0 + DepMap 谱系依赖。
- **D4：** D0 仅把 C3 拆成 MIR 与 TFT；增加随机拆分的容量对照。
- **D5：** D0 仅替换高质量 PPI；增加边数或度分布控制。

预算紧时先跑 D0、D1、D4。D4 的准备成本低，但需要先完成第十条视图的加载改动。三组先用同一个种子跑完整五折，有价值的方案再共同使用另外 2–4 个种子复核。

### 第二轮：根据第一轮结果选择

- **D6：** 添加稳定的癌种相关共表达视图。
- **D7：** 在 C3 拆分基础上，再加有方向的 TF 调控边，验证独立增量。
- **D8：** 去冗余 / 更新通路和 GO；拆成两个实验，避免同时变化。
- **D9：** 添加蛋白组；磷酸化组单独留作后续。

不要一次组合所有有希望的数据。先取最稳定的两项，比较 D0、D0+A、D0+B、D0+A+B；若合并没有额外提升，应考虑冗余和噪声。

### 第三轮：检验新数据是否让逐基因权重更有价值

固定一项选中的数据增强 A，跑四个条件：

```text
旧数据 + E1 全局权重
旧数据 + E2 逐基因权重
新数据 + E1 全局权重
新数据 + E2 逐基因权重
```

比较：
```text
数据收益（global） = AP(新数据,E1) - AP(旧数据,E1)
数据收益（node）   = AP(新数据,E2) - AP(旧数据,E2)
配合项            = 数据收益（node） - 数据收益（global）
```

配合项稳定为正，才支持“逐基因权重更适合这项数据增强”；单次为正不代表统计显著。之后再单独加入 E4 的中间融合。

### 九条视图、仅特征增强时的运行命令

以下命令以**已准备完成的** `data_variants/cnv_v1` 为前提，在服务器项目根目录执行。数据版本内必须含图、blocks、划分等与原项目相同的相对路径：

```bash
python run_ablation.py --experiments E1 E2 --seeds 42 --data-root . --devices 0 --output results/data_D0
python run_ablation.py --experiments E1 E2 --seeds 42 --data-root data_variants/cnv_v1 --devices 0 --output results/data_D1

python summarize_ablations.py results/data_D0 --output results/data_D0_summary
python summarize_ablations.py results/data_D1 --output results/data_D1_summary
```

现有汇总器会拒绝跨数据/代码哈希混合，也会拒绝重复的 E1/seed/fold。因此**不能把 D0 和 D1 两个目录一起传给它做数据对比**。先各自汇总 E1/E2；跨数据比较时按 `(架构, seed, fold)` 配对，并检查：

- 标签、划分、节点顺序、训练超参数一致。
- 只允许计划中的输入文件发生变化；把改变内容列入报告。
- 数据变更与模型代码变更分开记录。
- 新维度或新视图改变参数量时，补充同维/同容量对照。
- 若涉及标签更新，另建实验协议，不能作为“只加特征”的配对比较。

### 如何决定值得保留

> 先确定主要指标为验证 AP，并由同一验证最优 checkpoint 输出 AUC、Macro-F1 等；最终测试集只在方案锁定后使用。

至少同时观察：

- 每个种子内的五折平均 AP 差值、不同种子的方向与波动，是否由单折贡献大部分增益。
- 相同预测基因集合上的 AP/AUC；有覆盖与缺失子集的表现。
- 参数量、训练耗时、峰值显存、数据清洗维护成本。
- Top-K 的验证证据是否真正独立；验证集上挑选的候选不能再当作独立证据。
- 新视图的逐基因 beta 是否稳定，是否只是追踪节点度数、数据缺失或来源偏差。

可以预先约定例如 AP 绝对提升 0.003 的实际价值门槛，但这是预算标准，不能替代统计证据。五折训练集重叠，不把所有折当独立样本做普通 t 检验。注意力权重不是因果解释。

## 6. 首轮下载与交付清单

- [ ] 固定现有 gene_toids、labels、blocks 和图文件的版本；整理 HGNC 映射。
- [ ] 选择同一研究批次的 16 癌种基因级 CNA 表和样本表；不必先下载 FASTQ/BAM。
- [ ] 计算每基因、每癌种扩增/缺失频率，输出 32 列及缺失/样本计数。
- [ ] 使用本地 C3 JSON 生成 MIR/TFT 清单，单独完成视图拆分实验。
- [ ] 检查现有突变原始数据能否复用；不足时再下载开放 MAF。
- [ ] 选择一个固定的 DepMap release，下载依赖矩阵与模型元数据。
- [ ] 每项记录字段、映射率、来源、处理参数和 SHA256。
- [ ] 服务器验证输入尺寸、缓存对应、缺失处理后，再进行完整五折。

**暂不优先：** 原始测序文件、全量单细胞矩阵、影像、药物敏感性、长链非编码 RNA / enhancer 的全面建图。这些并非无用，而是当前基因级任务尚需定义对应标签、映射和对照；比 CNV、C3 整理与功能依赖更容易扩大工作量。

本次交付仅新增说明文档与交互清单，并补充 EXPERIMENTS.md 的逐基因权重解释。没有下载实验数据、安装依赖或运行本地训练。

