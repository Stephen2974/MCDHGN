$ErrorActionPreference = "Stop"
# Prefer free-GPU queue: do not wait for a whole wave before filling idle cards.
& "G:\LLLM\MCDHGN-main\scripts\run_ablation_queue.ps1" `
    -Experiments E2 `
    -Seeds 42 `
    -Folds 0,1,2,3,4 `
    -Devices 0,1,2 `
    -DataRoot "." `
    -Output "results\p0_legacy_D0" `
    -SummarizeOutput "results\p0_legacy_D0_summary"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
