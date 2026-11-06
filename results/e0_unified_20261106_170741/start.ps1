$ErrorActionPreference = "Stop"
# Free-GPU queue for unified-protocol E0 (seeds 42/43/44 x folds 0-4).
# Do not stop other jobs; do not overwrite p0_legacy_D0.
& "G:\LLLM\MCDHGN-main\scripts\run_ablation_queue.ps1" `
    -Experiments E0 `
    -Seeds 42,43,44 `
    -Folds 0,1,2,3,4 `
    -Devices 0,1,2 `
    -DataRoot "." `
    -Output "results\e0_unified_20261106_170741" `
    -SummarizeOutput "results\e0_unified_20261106_170741_summary" `
    -SkipSummarize
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
