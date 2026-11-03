$ErrorActionPreference = "Stop"
Set-Location G:\LLLM\MCDHGN-main
$env:DGLBACKEND = "pytorch"
$env:PYTHONUNBUFFERED = "1"
$py = "G:\LLLM\MCDHGN-main\.conda\python.exe"
New-Item -ItemType Directory -Force -Path results\logs | Out-Null

function Start-Fold([int]$Fold, [int]$Gpu) {
    $p = Start-Process -FilePath $py -WorkingDirectory "G:\LLLM\MCDHGN-main" -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "results\logs\e2_fold$Fold.log" `
        -RedirectStandardError "results\logs\e2_fold$Fold.err" `
        -ArgumentList @(
            "run_ablation.py", "--experiments", "E2", "--seeds", "42", "--folds", "$Fold",
            "--data-root", ".", "--devices", "$Gpu", "--output", "results/p0_legacy_D0"
        )
    Write-Host "fold $Fold -> GPU $Gpu  pid $($p.Id)"
    return $p
}

function Wait-Folds($Procs) {
    $Procs | Wait-Process
    $Procs | ForEach-Object { $_.Refresh() }
    $failed = @($Procs | Where-Object { $_.ExitCode -ne 0 })
    if ($failed) {
        $failed | ForEach-Object { Write-Host "失败 pid $($_.Id) exit $($_.ExitCode)，查看 results\logs\e2_fold*.err" }
        exit 1
    }
}

Write-Host "E2 runner started $(Get-Date -Format o)"
Wait-Folds @(
    (Start-Fold 0 0),
    (Start-Fold 1 1),
    (Start-Fold 2 2)
)
Write-Host "wave1 complete $(Get-Date -Format o)"
Wait-Folds @(
    (Start-Fold 3 0),
    (Start-Fold 4 1)
)
Write-Host "wave2 complete $(Get-Date -Format o)"
& $py summarize_ablations.py results/p0_legacy_D0 --output results/p0_legacy_D0_summary
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "E2 runner finished $(Get-Date -Format o)"
