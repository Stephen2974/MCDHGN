# D1 CNV 执行日志

时间使用 Asia/Shanghai。只记录已经执行的步骤。

## 2026-11-03T16:27:26+08:00 · 保护文件检查 · passed

开始前核对 113 个 D0 输入和 P0 封存文件。

## 2026-11-03T16:40:09+08:00 · 下载 · completed

提交 860b63a7f2d23540c94e25dd35302e133c29f48b；文件 120 个；LFS 端点已记录且未保存签名 URL。

## 2026-11-03T16:40:11+08:00 · all · failed

kirc_tcga_pan_can_atlas_2018 type_of_cancer is ccrcc

Traceback (most recent call last):
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 563, in main
    build(info)
    ~~~~~^^^^^^
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 516, in build
    store, samples, issues, links, studies = process_sources(OUT / "sources", gene_rows, index)
                                             ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 175, in process_sources
    raise ValueError(f"{study} type_of_cancer is {study_meta['type_of_cancer']}")
ValueError: kirc_tcga_pan_can_atlas_2018 type_of_cancer is ccrcc

## 2026-11-03T16:41:30+08:00 · 保护文件检查 · passed

开始前核对 113 个 D0 输入和 P0 封存文件。

## 2026-11-03T16:41:37+08:00 · 处理 kirc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"KIRC": 509}, "exclusions": {"no_cna_data": 3}, "mapped_rows": 12921, "conflicts": 7, "illegal_cells": 0}

## 2026-11-03T16:41:49+08:00 · 处理 brca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"BRCA": 1070}, "exclusions": {"no_cna_data": 14}, "mapped_rows": 12921, "conflicts": 14, "illegal_cells": 0}

## 2026-11-03T16:41:56+08:00 · 处理 coadread_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"READ": 155, "COAD": 437}, "exclusions": {"no_cna_data": 2}, "mapped_rows": 12922, "conflicts": 10, "illegal_cells": 0}

## 2026-11-03T16:42:02+08:00 · 处理 prad_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"PRAD": 489}, "exclusions": {"no_cna_data": 5}, "mapped_rows": 12921, "conflicts": 16, "illegal_cells": 0}

## 2026-11-03T16:42:07+08:00 · 处理 stad_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"STAD": 438}, "exclusions": {"missing_clinical_record": 1, "no_cna_data": 2}, "mapped_rows": 12921, "conflicts": 13, "illegal_cells": 0}

## 2026-11-03T16:42:13+08:00 · 处理 hnsc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"HNSC": 517}, "exclusions": {"no_cna_data": 6}, "mapped_rows": 12921, "conflicts": 12, "illegal_cells": 0}

## 2026-11-03T16:42:19+08:00 · 处理 luad_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"LUAD": 511}, "exclusions": {"no_cna_data": 55}, "mapped_rows": 12921, "conflicts": 11, "illegal_cells": 0}

## 2026-11-03T16:42:25+08:00 · 处理 thca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"THCA": 497}, "exclusions": {"missing_clinical_record": 1, "no_cna_data": 3}, "mapped_rows": 12921, "conflicts": 5, "illegal_cells": 0}

## 2026-11-03T16:42:30+08:00 · 处理 blca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"BLCA": 408}, "exclusions": {"no_cna_data": 3}, "mapped_rows": 12921, "conflicts": 12, "illegal_cells": 0}

## 2026-11-03T16:42:32+08:00 · 处理 esca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"ESCA": 182}, "exclusions": {}, "mapped_rows": 12921, "conflicts": 9, "illegal_cells": 0}

## 2026-11-03T16:42:37+08:00 · 处理 lihc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"LIHC": 367}, "exclusions": {"no_cna_data": 5}, "mapped_rows": 12921, "conflicts": 11, "illegal_cells": 0}

## 2026-11-03T16:42:43+08:00 · 处理 ucec_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"UCEC": 523}, "exclusions": {"missing_clinical_record": 1, "no_cna_data": 6}, "mapped_rows": 12921, "conflicts": 11, "illegal_cells": 0}

## 2026-11-03T16:42:48+08:00 · 处理 lusc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"LUSC": 487}, "exclusions": {}, "mapped_rows": 12921, "conflicts": 10, "illegal_cells": 0}

## 2026-11-03T16:42:52+08:00 · 处理 cesc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"CESC": 293}, "exclusions": {"no_cna_data": 4}, "mapped_rows": 12921, "conflicts": 8, "illegal_cells": 0}

## 2026-11-03T16:42:55+08:00 · 处理 kirp_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"KIRP": 283}, "exclusions": {}, "mapped_rows": 12921, "conflicts": 3, "illegal_cells": 0}

## 2026-11-03T16:42:58+08:00 · build · failed

name 'IDENTITY' is not defined

Traceback (most recent call last):
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 569, in main
    build()
    ~~~~~^^
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 528, in build
    report = write_outputs(store, gene_rows, samples, issues, links, studies, identity, source_info)
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 394, in write_outputs
    emit_tsv(OUT / "cnv_features.tsv", features, IDENTITY + columns)
                                                 ^^^^^^^^
NameError: name 'IDENTITY' is not defined

## 2026-11-03T16:43:27+08:00 · 保护文件检查 · passed

开始前核对 113 个 D0 输入和 P0 封存文件。

## 2026-11-03T16:43:35+08:00 · 处理 kirc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"KIRC": 509}, "exclusions": {"no_cna_data": 3}, "mapped_rows": 12921, "conflicts": 7, "illegal_cells": 0}

## 2026-11-03T16:43:49+08:00 · 处理 brca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"BRCA": 1070}, "exclusions": {"no_cna_data": 14}, "mapped_rows": 12921, "conflicts": 14, "illegal_cells": 0}

## 2026-11-03T16:43:57+08:00 · 处理 coadread_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"READ": 155, "COAD": 437}, "exclusions": {"no_cna_data": 2}, "mapped_rows": 12922, "conflicts": 10, "illegal_cells": 0}

## 2026-11-03T16:44:04+08:00 · 处理 prad_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"PRAD": 489}, "exclusions": {"no_cna_data": 5}, "mapped_rows": 12921, "conflicts": 16, "illegal_cells": 0}

## 2026-11-03T16:44:10+08:00 · 处理 stad_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"STAD": 438}, "exclusions": {"missing_clinical_record": 1, "no_cna_data": 2}, "mapped_rows": 12921, "conflicts": 13, "illegal_cells": 0}

## 2026-11-03T16:44:17+08:00 · 处理 hnsc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"HNSC": 517}, "exclusions": {"no_cna_data": 6}, "mapped_rows": 12921, "conflicts": 12, "illegal_cells": 0}

## 2026-11-03T16:44:24+08:00 · 处理 luad_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"LUAD": 511}, "exclusions": {"no_cna_data": 55}, "mapped_rows": 12921, "conflicts": 11, "illegal_cells": 0}

## 2026-11-03T16:44:31+08:00 · 处理 thca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"THCA": 497}, "exclusions": {"missing_clinical_record": 1, "no_cna_data": 3}, "mapped_rows": 12921, "conflicts": 5, "illegal_cells": 0}

## 2026-11-03T16:44:36+08:00 · 处理 blca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"BLCA": 408}, "exclusions": {"no_cna_data": 3}, "mapped_rows": 12921, "conflicts": 12, "illegal_cells": 0}

## 2026-11-03T16:44:39+08:00 · 处理 esca_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"ESCA": 182}, "exclusions": {}, "mapped_rows": 12921, "conflicts": 9, "illegal_cells": 0}

## 2026-11-03T16:44:44+08:00 · 处理 lihc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"LIHC": 367}, "exclusions": {"no_cna_data": 5}, "mapped_rows": 12921, "conflicts": 11, "illegal_cells": 0}

## 2026-11-03T16:44:51+08:00 · 处理 ucec_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"UCEC": 523}, "exclusions": {"missing_clinical_record": 1, "no_cna_data": 6}, "mapped_rows": 12921, "conflicts": 11, "illegal_cells": 0}

## 2026-11-03T16:44:59+08:00 · 处理 lusc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"LUSC": 487}, "exclusions": {}, "mapped_rows": 12921, "conflicts": 10, "illegal_cells": 0}

## 2026-11-03T16:45:04+08:00 · 处理 cesc_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"CESC": 293}, "exclusions": {"no_cna_data": 4}, "mapped_rows": 12921, "conflicts": 8, "illegal_cells": 0}

## 2026-11-03T16:45:09+08:00 · 处理 kirp_tcga_pan_can_atlas_2018 · completed

{"selected_patients": {"KIRP": 283}, "exclusions": {}, "mapped_rows": 12921, "conflicts": 3, "illegal_cells": 0}

## 2026-11-03T16:45:14+08:00 · 构建特征表 · data_prepared_not_trainable

缺失单元格 1860。身份未解析节点 5 个，状态来自 P0 gene_map 的实际 mapping_status。 原 48 维矩阵没有患者 ID，无法核实与本批 CNA 患者是否完全相同。基因组版本未写在 meta_cna.txt 中，因此记录为未知，不从其他文件名推断。 KIRC、KIRP、LIHC、THCA 的 meta type_of_cancer 分别是 ccrcc、prcc、hcc、thpa；研究 ID 仍是 TCGA 项目缩写。 ESCA 研究显示名是 Esophageal Adenocarcinoma，本列仍按该研究中的原发 CNA 样本汇总，没有再按组织学拆开。

## 2026-11-03T16:46:36+08:00 · 质量复查 · data_prepared_not_trainable

缺失单元格 1860；保护文件 113 个未变化。

## 2026-11-03T16:48:17+08:00 · 说明更新 · completed

根据已生成的质量报告补充缺失构成，并更新清单里的代码与 README 哈希。没有重新下载，也没有改动频率值。

## 2026-11-03T23:30:47+08:00 · quality · failed

cannot access local variable 'rows' where it is not associated with a value

Traceback (most recent call last):
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 711, in main
    quality()
    ~~~~~~~^^
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 678, in quality
    features = read_d1_tsv(OUT / "cnv_features.tsv")
  File "D:\MCDHGN\data_processing\d1\tables.py", line 37, in read_d1_tsv
    rows.append(dict(zip(header, parts)))
    ^^^^
UnboundLocalError: cannot access local variable 'rows' where it is not associated with a value

## 2026-11-03T23:30:58+08:00 · quality · failed

sample_metadata has a missing or repeated sample_id

Traceback (most recent call last):
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 711, in main
    quality()
    ~~~~~~~^^
  File "D:\MCDHGN\data_processing\d1\pipeline.py", line 684, in quality
    check = check_tables(features, observed, patients, events, gene_rows, reasons=reasons, sample_rows=sample_rows)
  File "D:\MCDHGN\data_processing\d1\tables.py", line 243, in check_tables
    limits = selected_patient_limits(sample_rows) if sample_rows is not None else None
             ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "D:\MCDHGN\data_processing\d1\tables.py", line 208, in selected_patient_limits
    raise ValueError("sample_metadata has a missing or repeated sample_id")
ValueError: sample_metadata has a missing or repeated sample_id

## 2026-11-03T23:31:24+08:00 · 质量复查 · data_prepared_not_trainable

缺失单元格 1860；保护文件 113 个未变化。

## 2026-11-03T23:31:51+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 113, "mismatches": 0}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-03T23:33:57+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 0, "mismatches": 0, "scope": "not_checked_for_temporary_copy"}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-03T23:34:02+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 113, "mismatches": 0}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-03T23:34:13+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 0, "mismatches": 0, "scope": "not_checked_for_temporary_copy"}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-03T23:34:40+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 113, "mismatches": 0}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-03T23:34:51+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 0, "mismatches": 0, "scope": "not_checked_for_temporary_copy"}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-03T23:35:02+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 0, "mismatches": 0, "scope": "not_checked_for_temporary_copy"}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-03T23:35:21+08:00 · 离线完整性验证 · passed

{"status": "data_prepared_not_trainable", "trainable": false, "missing_cells": 1860, "protected": {"checked": 113, "mismatches": 0}, "sources": {"files": 120, "lfs_files": 45}, "outputs": 15, "code_checked": 11, "code_changed_since_manifest": [], "checked_pairs": 207104}

## 2026-11-05T22:05:49+08:00 · 保护文件与冻结身份检查 · passed

开始前核对 113 个 D0/P0 保护文件、P0 映射及 HGNC 快照。
