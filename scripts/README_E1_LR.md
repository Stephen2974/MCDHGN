# E1 学习率对比：0.01、0.005、0.003

E1 固定为 GATv2 + 全局元路径权重 + 无跨视图交互。三组都使用 seed 42
的完整五折，共 15 次全新正式训练，记录中的 0.01 也重新训练。
其余设置从 `results/p0_legacy_D0/E1/seed42` 逐项读取：数据、划分、
epochs、eval_every、patience、weight_decay、positive_weight、dropout、
hidden_size、heads、fusion_dim、fusion_chunk_size、优化器 Adam。
不增加学习率 scheduler，不修改模型或训练代码，不升级依赖，不调用测试集。

达到记录中的 1000 轮上限后按原早停和上限结束，不自动延长。
每张卡同时只训练一折。正式调度与既有 E3 启动器相同，并对每个学习率重复一次：

- GPU0：fold0，完成后 fold3。
- GPU1：fold1，完成后 fold4。
- GPU2：fold2。

三个学习率按 0.01、0.005、0.003 顺序各跑完五折。某一折失败时停止本次其余 worker，
保留已写出的目录和日志，不自动改参数重跑。

## 环境

使用此前成功运行 E1 的项目内环境，不要新建或升级：

```text
G:\LLLM\MCDHGN-main\.conda\python.exe
Python 3.9.16
PyTorch 1.12.0+cu116
DGL 1.1.1+cu116
CUDA runtime 11.6
三张 NVIDIA GeForce RTX 2080 Ti
```

若 GPU 上仍有 Python 训练进程，启动器每 60 秒查询一次并等待，不结束那些进程。
Windows 会把资源管理器、浏览器和 Cursor 也列在 compute 进程里；这些桌面进程不阻塞启动。

## 文件校验与启动器测试

在项目根目录执行。`--check-files-only` 只核对五折 E1 配置和当前代码/数据哈希，不检查 CUDA。

```powershell
Set-Location G:\LLLM\MCDHGN-main
$env:DGLBACKEND = "pytorch"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "0,1,2"
$py = ".\.conda\python.exe"

& $py -m data_processing.p0.release verify
if ($LASTEXITCODE -ne 0) { throw "P0 integrity check failed; do not alter manifests" }
& $py scripts\run_e1_lr_sweep.py --check-files-only
if ($LASTEXITCODE -ne 0) { throw "E1 files/configuration/hash check failed" }
& $py -m unittest discover -s scripts -p test_e1_lr_sweep.py -v
if ($LASTEXITCODE -ne 0) { throw "Launcher tests failed" }
```

不要修改 `data_versions/p0_v1` 的封存清单来让校验通过。P0 校验会追加审计日志，
不改封存对象。校验失败时保留现场，不要改哈希、模型或训练参数后重跑。

## 正式启动

不要把 `--output` 指到已有结果目录。省略 `--output` 时，启动器新建
`results\e1_lr_sweep_yyyyMMdd_HHmmss`。

```powershell
& $py scripts\run_e1_lr_sweep.py
if ($LASTEXITCODE -ne 0) { throw "E1 learning-rate sweep failed; retain the new results directory and feedback archive" }
```

启动后的顺序是：新建目录、复制既有 0.01 的五折对照文件、等待 GPU 空闲、
P0 完整性校验、三卡环境校验、项目 `tests`、真实图校验、GPU0 上真实数据和
完整模型尺寸的两轮 E1 试跑，然后才开始 15 次正式训练。试跑只把
epochs/eval_every/patience 改为 2/1/0，写在 `probe/`，不进入汇总。
正式训练重新初始化模型和 `seed + 1000 * fold`。

## 结果

- `train/lr0.01|lr0.005|lr0.003/E1/seed42/fold0..4/`：正式配置、结果、history、预测、checkpoint、权重。
- `reference_lr001/`：本次读取的既有 0.01 快照。
- `comparison/summary.json`：三组均值和折间样本标准差、相对本轮 0.01 的配对差值与胜出折数、1000 轮上限、新旧 0.01 复现差异。
- `comparison/runs.csv`、`comparison/comparison.txt`：逐折记录和可读汇总。
- `logs/`、`run_status.json`、`graph_validation_server.json`：检查和训练日志。

正常结束或捕获到失败后，生成同级 `results\e1_lr_sweep_<时间>_feedback.tar.gz`。
`best.pt` 与 `gene_metapath_weights.npz` 不进入压缩包，保留在结果目录。
强制结束进程或断电可能来不及打包，此时保留整个输出目录。
