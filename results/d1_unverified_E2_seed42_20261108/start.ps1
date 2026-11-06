$ErrorActionPreference = "Stop"
# Free-GPU queue: as soon as a fold finishes, the next fold starts on that GPU.
# Do not wait for wave1 (folds 0-2) before launching folds 3-4.
& "G:\LLLM\MCDHGN-main\scripts\run_ablation_queue.ps1" `
    -Experiments E2 `
    -Seeds 42 `
    -Folds 0,1,2,3,4 `
    -Devices 0,1,2 `
    -DataRoot "data_variants\d1_cnv_unverified_v1" `
    -Output "results\d1_unverified_E2_seed42_20261108" `
    -SummarizeOutput "results\d1_unverified_E2_seed42_20261108_summary"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
