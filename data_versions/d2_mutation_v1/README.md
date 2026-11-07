# d2_mutation_v1

D2 突变类型特征发布目录（审核用）。由 `python -m data_processing.d2.pipeline` 生成。

## 状态

构建完成后查看：

- `quality_report.json`：覆盖率、各癌种 callable 患者数、是否宣称可训练
- `protected_files_before.json` / `protected_files_after.json`：训练输入与代码哈希
- `braf_trace_example.json`：BRAF 真实追溯（优先 THCA）
- `REVIEW_CHECKLIST.md`

## 关键交付

- `mutation_type_features.tsv`：12,944 × 48（`MUT_{MISSENSE,TRUNC,SPLICE}:{cancer}`）
- `feature_schema.tsv` / `feature_observed.tsv`
- `mutation_patient_counts.tsv` / `callable_patient_counts.tsv`
- `sample_metadata.tsv`
- `gene_map.tsv` / `mapping_issues.tsv`
- `classification_map.tsv` / `sample_selection_and_dedup_rules.json`
- `recurrence/`：位点复发与候选集中度（独立，未拼入 48 列）
- `audit/gene_exon_lengths_gencode_v28.tsv`：长度审计，未用于频率

后续训练接入方案（待审核，本次未执行）：原 48 维 + 本 48 维 = 96 维；需显式缺失协议后另建 `data_variants`。
