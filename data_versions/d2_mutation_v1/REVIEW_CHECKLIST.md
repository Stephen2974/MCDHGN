# D2 人工审核清单

- [ ] 确认旧 GDC Mutect2 UUID 不可用，改用 DataHub 提交 860b63a7f2d23540c94e25dd35302e133c29f48b
- [ ] 确认分母来自 cases_sequenced，且文档写明非逐基因 callable
- [ ] 确认未做基因长度校正、未中位数插补、未写入旧 mut_data
- [ ] 确认 U2AF1L5/CBSL/TAZ/MYO18A/TIAF1 无自动填值
- [ ] 确认 recurrence/ 未拼入 mutation_type_features 的 48 列
- [ ] 核对 BRAF 追溯案例 braf_trace_example.json
- [ ] 核对 protected_files_before/after 哈希一致
- [ ] 未启动训练、未改默认 data-root
