# Windows：E3 seed42 五折，三张 RTX 2080 Ti 并行

E3 = GATv2 + 全局元路径权重 + 跨视图交互。相对 E1，仅将
`fusion_mode=none` 改为 `cross`；不是 E2 的逐基因权重。
使用独立启动器，不修改模型实现、数据或训练脚本。此次将汇总脚本恢复为
精确匹配 P0 哈希的版本，保留已有保护逻辑在独立入口；不修改封存清单。

## P0 哈希修复说明

原提交 `485dcb0` 给 `summarize_ablations.py` 增加了旧汇总目录保护，且本地
文件使用 CRLF，因此与封存哈希不符。恢复 `f92bfcb` 中的原始 LF 字节后，
SHA256 为 `eecd73da86bf3ea9c97301d6db2d05c9af0a92e2d7f454dae658c2bca2d123be`。
`.gitattributes` 为该文件指定 `eol=lf`，避免 Windows checkout 再转换。
原修改的完整字节已备份在开发电脑
`results/p0_hash_repair_20261103_144221/summarize_ablations_before.py`，
同目录的 `repair_record.json` 记录备份及封存清单哈希。

旧汇总保护行为保留在 `scripts/summarize_ablations_safe.py`，支持原参数：
若输出目录名为 `p0_legacy_D0_summary` 且已有 summary.json，则改写到
`p0_legacy_D0_summary_seeds42_43_44`。以后需要此保护行为时使用该入口。
E3 自动汇总使用新建且独立的比较目录，不涉及旧汇总目录。

## 1. 同步文件

本次同步七个文件：`.gitattributes`、`summarize_ablations.py`、
`scripts/run_e3_parallel.py`、`scripts/test_e3_parallel.py`、
`scripts/README_E3.md`、`scripts/summarize_ablations_safe.py`、
`scripts/test_summary_safe.py`。
启动器独立运行，不依赖 E2 启动器。

在开发电脑提交（不包含其他未提交修改）：

```powershell
Set-Location D:\MCDHGN
$syncFiles = @(
    ".gitattributes", "summarize_ablations.py",
    "scripts/run_e3_parallel.py", "scripts/test_e3_parallel.py", "scripts/README_E3.md",
    "scripts/summarize_ablations_safe.py", "scripts/test_summary_safe.py"
)
git add -- $syncFiles
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --name-status
git commit --only -m "Restore P0 summary integrity and add E3 seed42 launcher" -- $syncFiles
if ($LASTEXITCODE -ne 0) { throw "Commit failed" }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
```

在实验电脑同步：

```powershell
Set-Location D:\MCDHGN
git status --short
git -c core.autocrlf=false pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw "Pull failed; preserve local work and resolve before training" }
```

不要使用 `git reset --hard` 或 `git clean`。这里临时关闭自动换行转换，
后续仍必须通过实际 SHA256 校验；Git 同步成功本身不能证明 P0 完整性。

实验电脑必须保留跑 E1 时的完整项目、原始数据、图、标签和划分，
以及 `data_versions/p0_v1` 的校验材料。不要重新生成图、标签或划分。
`results/` 被 Git 忽略，需要在实验电脑原位保留或单独复制：

```text
results/p0_legacy_D0/E1/seed42/fold0/
results/p0_legacy_D0/E1/seed42/fold1/
results/p0_legacy_D0/E1/seed42/fold2/
results/p0_legacy_D0/E1/seed42/fold3/
results/p0_legacy_D0/E1/seed42/fold4/
```

每折至少包含 `config.json`、`result.json`、`history.jsonl`、
`validation_predictions.npz`；原有 `best.pt` 也应保留归档。
默认基线目录精确指向 `E1/seed42`，其他种子和 E2 结果不会参与本次比较。

## 2. 检查并启动

在 Anaconda PowerShell Prompt 执行。环境名替换为原来成功跑 E1 的环境；
项目若不在 D 盘，只修改 `Set-Location`。不要新建或升级训练环境。
E1 记录为 Python 3.9.16、PyTorch 1.12.0+cu116、DGL 1.1.1+cu116。
启动器还会严格检查 Python build、CUDA runtime 及 GPU 型号。

```powershell
Set-Location D:\MCDHGN
conda activate YOUR_EXISTING_E1_ENVIRONMENT
if ($LASTEXITCODE -ne 0) { throw "Environment activation failed" }
$env:DGLBACKEND = "pytorch"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0,1,2"

python -c "import sys,torch,dgl; print(sys.executable); print(sys.version); print(torch.__version__,dgl.__version__,torch.version.cuda); print('GPUs:',torch.cuda.device_count())"
if ($LASTEXITCODE -ne 0) { throw "Environment import failed" }
nvidia-smi

python -m data_processing.p0.release verify
if ($LASTEXITCODE -ne 0) { throw "P0 integrity check failed; do not alter manifests" }
python scripts/run_e3_parallel.py --check-files-only
if ($LASTEXITCODE -ne 0) { throw "E1 files/configuration/hash check failed" }
python -m unittest discover -s scripts -p test_e3_parallel.py -v
if ($LASTEXITCODE -ne 0) { throw "Launcher tests failed" }
python -m unittest discover -s scripts -p test_summary_safe.py -v
if ($LASTEXITCODE -ne 0) { throw "Summary compatibility test failed" }

$runDir = Join-Path (Get-Location).Path ("results\d0_E3_seed42_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
python scripts/run_e3_parallel.py --output "$runDir"
if ($LASTEXITCODE -ne 0) { throw "E3 failed; inspect $runDir\logs and retain the feedback archive" }

Get-Content "$runDir\comparison\comparison.txt"
Write-Host "Return archive: ${runDir}_feedback.tar.gz"
```

`--check-files-only` 校验五折 E1 配置和当前数据/代码哈希，不检查 CUDA。
实际启动后按顺序执行：P0 完整性校验、三卡 DGL 运算及环境校验、
项目 `tests`、真实图验证、GPU0 上真实数据和完整模型尺寸的两轮 E3 试跑，
通过后再开始正式五折。试跑仅把 epochs/eval_every/patience 改为 2/1/0，
单独保存在 `probe/`，不参与汇总；正式训练重新初始化模型与随机种子。
P0 校验会正常追加审计日志，不修改封存对象和清单。

五折调度（每张卡最多同时一折，每个进程仅用一张卡）：

- GPU0：fold0 完成后跑 fold3。
- GPU1：fold1 完成后跑 fold4。
- GPU2：fold2。

fold3 不必等 GPU1/2 完成即可开始；fold4 同理。
不同折重新创建模型和优化器，不继承前一折训练状态。
base seed 为 42；沿用原训练脚本 `seed + 1000 * fold` 的种子规则。

正式训练从 E1 记录逐项读取：epochs=1000、eval_every=10、patience=20、
lr=0.01、weight_decay=0.0001、positive_weight=1.904、dropout=0.4、
hidden_size=256、heads=4、fusion_dim=64、fusion_chunk_size=1024。
fusion_alpha 仍为 0.6，保留原验证 AP 选最佳 checkpoint 的协议。
epochs=1000 是上限，仍按原规则早停。

11 GB 是每张卡独立可用的容量，不能合并成单折 33 GB。
E3 新增交互模块，显存需求需由试跑确认；启动器不会为避免 OOM 自动降低
hidden_size、heads、fusion_dim 或 chunk_size。试跑不能保证长训练和三进程的
系统内存一定充足：需给三个图/数据副本预留 RAM，并关闭其他 GPU 工作负载。
保持启动窗口开启、电脑不睡眠。若失败，不要先改参数重跑后再与 E1 当作同协议比较。

基线放在其他目录时，可显式指定，但必须只包含 E1 seed42 的五折：

```powershell
python scripts/run_e3_parallel.py --baseline "E:\existing_results\E1\seed42" --check-files-only
# 校验通过后：
python scripts/run_e3_parallel.py --baseline "E:\existing_results\E1\seed42"
```

## 3. 监控、自动比较与回传

另一 PowerShell 窗口查看 GPU；Ctrl+C 只结束该监控命令：

```powershell
nvidia-smi -l 2
```

另一窗口查看日志（将 ACTUAL_TAG 替换为启动时打印的实际目录名）：

```powershell
Get-Content "D:\MCDHGN\results\d0_E3_seed42_ACTUAL_TAG\logs\gpu0.log" -Tail 30 -Wait
```

GPU1、GPU2 日志同理；正式训练前查看 `logs/05_real_E3_probe.log`。
完成后自动检查正式 E3 的参数、数据指纹、环境与对应 E1，且只汇总
`reference_E1/` 和 `train/`。比较方向固定为 E3−E1。

结果目录内容：

- `train/E3/seed42/fold0..4/`：正式配置、结果、history、预测、checkpoint、权重。
- `reference_E1/E1/seed42/fold0..4/`：本次使用的 E1 比较输入快照。
- `comparison/summary.json`：五折均值、标准差、各折配对差值和胜出折数。
- `comparison/runs.csv`：E1/E3 共十条正式运行记录。
- `comparison/comparison.txt`：AP、AUC、accuracy、Macro-F1 的平均差值（百分点）和胜出折数。
- `logs/`、`graph_validation_server.json`、`run_status.json`：检查、训练日志及准确启动命令。
- `probe/`：两轮试跑记录，排除在正式比较之外。

`summary.json` 中对应 contrast 是 `cross_view_with_global_weights`。
主要看 AP，结合 AUC、Macro-F1、逐折差值及计算成本；seed42 只提供首轮筛选证据。
若后续要判断收益稳定性，需保持同协议再补 E3 seed43、44，与已有 E1 配对。

正常完成或捕获到失败后，自动生成同级 `${runDir}_feedback.tar.gz`。
**回传该压缩包即可**，内含日志、对比、配置、结果、预测和历史；
`best.pt` 与 `gene_metapath_weights.npz` 为节省体积不入包，请在实验电脑保留。
E3 权重仍为全局权重，本次主要分析不依赖逐基因权重文件。
强制结束进程、断电可能来不及打包，届时保留整个输出目录。

任何 worker 失败会停止本次其他 worker，保留部分结果并记录失败，不输出完整比较。
本启动器不做断点续训；重新启动使用新目录，会从头跑全部五折。
不要重复使用已有输出目录，也不要将部分结果混入完整结果。

需要单独重新生成比较时（已确认五折全部成功）：

```powershell
$runDir = "D:\MCDHGN\results\d0_E3_seed42_ACTUAL_TAG"
python summarize_ablations.py "$runDir\reference_E1" "$runDir\train" --expected-folds 5 --output "$runDir\comparison"
if ($LASTEXITCODE -ne 0) { throw "Comparison failed" }
```

不要对整个 `$runDir` 或 `results/` 递归汇总：其中有试跑、其他种子或重复快照。
