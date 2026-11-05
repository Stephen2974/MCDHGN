# D1：TCGA PanCancer Atlas 2018 离散 GISTIC

在项目根目录执行。这个流程只生成 `data_versions/d1_cnv_v1/` 的频率、掩码和审计文件，不改 D0 图、P0 封存文件、标签、划分或训练参数，也不启动训练。

```powershell
python -m data_processing.d1.pipeline verify
python -m unittest tests.test_d1
python -m data_processing.d1.build_graph --features data_versions/d1_cnv_v1/cnv_features.tsv
```

`pipeline verify` 只读核对固定提交、来源文件、登记输出、处理代码、P0 映射和 HGNC 快照，不联网也不更新 manifest。`build_graph` 在仍有缺失或没有显式缺失处理协议时拒绝写入 `data_variants/cnv_v1`；只有另行提供协议并生成有限矩阵后，才可临时构建并打包 80 列图。

实际来源、癌种患者数、缺失和完成状态写在 `data_versions/d1_cnv_v1/README.md` 与 `quality_report.json`。执行记录在 `data_processing/d1/EXECUTION_LOG.md`。

固定来源是 cBioPortal DataHub 提交 `860b63a7f2d23540c94e25dd35302e133c29f48b` 的 TCGA PanCancer Atlas 2018 研究。提交快照和研究批次分开记录。CNA 必须是 `DISCRETE` / GISTIC；连续拷贝数文件不下载、不混用。COAD 和 READ 从 `coadread_tcga_pan_can_atlas_2018` 按 ONCOTREE 与部位规则拆开。

原始矩阵在 `data_versions/d1_cnv_v1/sources/`，该目录有自己的忽略规则，避免把大文件自动纳入 Git。根目录 `.gitignore` 未改。
