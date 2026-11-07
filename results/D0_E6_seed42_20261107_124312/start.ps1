$ErrorActionPreference = "Stop"
Set-Location "G:\LLLM\MCDHGN-main"
& "G:\LLLM\MCDHGN-main\scripts\run_ablation_queue.ps1" `
    -Experiments E6 `
    -Seeds 42 `
    -Folds 0,1,2,3,4 `
    -Devices 0,1,2 `
    -DataRoot "." `
    -Output "results\D0_E6_seed42_20261107_124312" `
    -SummarizeOutput "results\D0_E6_seed42_20261107_124312_summary" `
    -Python "G:\LLLM\MCDHGN-main\.conda\python.exe" `
    -ProjectRoot "G:\LLLM\MCDHGN-main" `
    -Epochs 1000 `
    -EvalEvery 10 `
    -Patience 20 `
    -Lr 0.01 `
    -WeightDecay 0.0001 `
    -PositiveWeight 1.904 `
    -Dropout 0.4 `
    -HiddenSize 256 `
    -Heads 4 `
    -FusionDim 64 `
    -FusionChunkSize 1024
exit $LASTEXITCODE
