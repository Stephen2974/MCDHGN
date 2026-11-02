# 来源核查记录

核查日期：2026-10-23。此文件记录可确认的信息和未解决项，不推断缺失的历史授权或证据日期。

## HGNC

官方[下载说明](https://hgnc.genenames.org/download/)给出 Google Cloud Storage 的完整 TSV 地址；[字段与历史快照说明](https://hgnc.genenames.org/download/archive/)说明 HGNC ID、现行/旧符号及其他数据库 ID 的含义。

本次使用官方下载文件，固定 SHA256、下载时间、HTTP Last-Modified 和 GCS generation。文件没有传统数字 release，按下载快照及内容哈希标识，不写“最新版”。官方[数据使用说明](https://www.genenames.org/about/)允许使用并请求注明 HGNC 来源。

## NCG7.0

本地目录及文件声明为 NCG7.0，包含癌症驱动记录、canonical/candidate 名单和正常组织 driver 文件。保留所有原文件哈希；没有证据证明历史下载日期，也尚未将本地文件与官方发布文件逐字节核对。

尝试访问官方[帮助](https://network-cancer-genes.org/help.php)和[下载页面](https://network-cancer-genes.org/download.php)，本次均超时。因此来源条件保持待核查，不能把网站不可用解释为允许或禁止某种用途。正常组织 driver 不自动当作癌症阴性或癌症阳性。

canonical 类型及 PMID 是原始表内容。没有自动获取或核查每篇论文的首次驱动证据；PMID 存在不等于已验证全文结论。

## OncoKB

本地 cancerGeneList.tsv 仅用于来源盘点和旧标签交集审计。本地文件的历史版本与授权未知。

截至核查日，官方[Licensing FAQ](https://faq.oncokb.org/licensing)说明不得将 OncoKB 用于 AI/ML 训练，验证/benchmark 需要明确许可。本轮不新增下载训练数据，也不让其决定新训练成员。旧标签构造是否使用过它无法从现有 notebook 证明，保持未知。

## CancerMine、ONGene 与未识别来源

CancerMine sentences 原表含 PMID 和论文日期；这些日期仅按原字段保留，不认为是首次驱动确认日期。文本挖掘提及不自动转为阳性。collated 文件保留原件及哈希，逐条审计使用 sentences 中的来源记录。

ONGene 按目录成员关系记录，未将成员资格自动视为 canonical 证据。CancerMine 与 ONGene 本地快照 release、历史下载时间和对应使用条件尚未核实。

mmc2.xlsx 的包内元数据记录作者名 Matthew Bailey、工作表 Table S2、2018 年创建时间；这些只是定位线索，不能据此认定原论文、正式 release 或证据时间。extract_name.txt 缺少逐基因原始来源。两者保留哈希并列入待识别清单，不据其自动改标签。

用户确认没有保留 cancermine_unfiltered.tsv、cancerneighbo.json、ctm.json、now_pos.pkl、now_neg.pkl 和上述 xlsx 的原论文信息。不能把当前 collated/sentences 文件冒充缺失的历史 unfiltered 文件。
