$ErrorActionPreference = "Stop"
Set-Location G:\LLLM\MCDHGN-main
$env:DGLBACKEND = "pytorch"
$env:PYTHONUNBUFFERED = "1"
$py = "G:\LLLM\MCDHGN-main\.conda\python.exe"
New-Item -ItemType Directory -Force -Path results\logs | Out-Null

$jobs = @()
foreach ($exp in @("E1", "E2")) {
    foreach ($seed in @(43, 44)) {
        foreach ($fold in 0..4) {
            $jobs += [pscustomobject]@{ Exp = $exp; Seed = $seed; Fold = $fold }
        }
    }
}

$gpus = @(0, 1, 2)
$running = @{}
$next = 0
$done = 0
$failed = @()

function Start-FoldJob($job, [int]$Gpu) {
    $tag = "{0}_s{1}_fold{2}" -f $job.Exp, $job.Seed, $job.Fold
    $proc = Start-Process -FilePath $py -WorkingDirectory "G:\LLLM\MCDHGN-main" -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "results\logs\$tag.log" `
        -RedirectStandardError "results\logs\$tag.err" `
        -ArgumentList @(
            "run_ablation.py", "--experiments", $job.Exp, "--seeds", "$($job.Seed)", "--folds", "$($job.Fold)",
            "--data-root", ".", "--devices", "$Gpu", "--output", "results/p0_legacy_D0"
        )
    Write-Host "START $tag -> GPU $Gpu pid $($proc.Id)"
    return $proc
}

Write-Host "seed43/44 runner started $(Get-Date -Format o) jobs=$($jobs.Count)"
while ($true) {
    $finished = @()
    foreach ($gpu in @($running.Keys)) {
        $item = $running[$gpu]
        $item.Proc.Refresh()
        if (-not $item.Proc.HasExited) { continue }
        $tag = "{0}_s{1}_fold{2}" -f $item.Job.Exp, $item.Job.Seed, $item.Job.Fold
        if ($item.Proc.ExitCode -ne 0) {
            Write-Host "FAILED $tag exit $($item.Proc.ExitCode)，查看 results\logs\$tag.err"
            $failed += $tag
        } else {
            $done++
            Write-Host "DONE $tag ($done/$($jobs.Count)) $(Get-Date -Format o)"
        }
        $finished += $gpu
    }
    foreach ($gpu in $finished) { $running.Remove($gpu) }

    if ($failed.Count -eq 0) {
        foreach ($gpu in $gpus) {
            if ($running.ContainsKey($gpu)) { continue }
            if ($next -ge $jobs.Count) { continue }
            $job = $jobs[$next]
            $next++
            $running[$gpu] = @{ Proc = (Start-FoldJob $job $gpu); Job = $job }
        }
    }

    if ($running.Count -eq 0 -and ($next -ge $jobs.Count -or $failed.Count -gt 0)) { break }
    Start-Sleep -Seconds 20
}

if ($failed.Count -gt 0) {
    Write-Host "FAILED tasks: $($failed -join ', ')"
    exit 1
}

Write-Host "二十次训练完成 $(Get-Date -Format o)"
& $py summarize_ablations.py results/p0_legacy_D0 --output results/p0_legacy_D0_summary_seeds42_43_44
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "seed43/44 runner finished $(Get-Date -Format o)"
