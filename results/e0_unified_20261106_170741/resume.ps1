$ErrorActionPreference = "Stop"
# Resume remaining E0 jobs only (completed folds are omitted).
& "G:\LLLM\MCDHGN-main\scripts\run_ablation_queue.ps1" `
    -Experiments E0 `
    -Seeds 42 `
    -Folds 3,4 `
    -Devices 0,1 `
    -DataRoot "." `
    -Output "results\e0_unified_20261106_170741" `
    -SkipSummarize
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& "G:\LLLM\MCDHGN-main\scripts\run_ablation_queue.ps1" `
    -Experiments E0 `
    -Seeds 43,44 `
    -Folds 0,1,2,3,4 `
    -Devices 0,1,2 `
    -DataRoot "." `
    -Output "results\e0_unified_20261106_170741" `
    -SummarizeOutput "results\e0_unified_20261106_170741_summary" `
    -SkipSummarize
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
