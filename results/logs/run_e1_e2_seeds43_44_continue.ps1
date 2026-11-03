$ErrorActionPreference = "Stop"
Set-Location G:\LLLM\MCDHGN-main
$env:DGLBACKEND = "pytorch"
$env:PYTHONUNBUFFERED = "1"
$py = "G:\LLLM\MCDHGN-main\.conda\python.exe"
New-Item -ItemType Directory -Force -Path results\logs | Out-Null

function Result-Path($job) {
    return "results/p0_legacy_D0/$($job.Exp)/seed$($job.Seed)/fold$($job.Fold)/result.json"
}

$runningNow = @{
    "E1:43:1" = $true
    "E1:43:2" = $true
}
$jobs = @()
foreach ($exp in @("E1", "E2")) {
    foreach ($seed in @(43, 44)) {
        foreach ($fold in 0..4) {
            $job = [pscustomobject]@{ Exp = $exp; Seed = $seed; Fold = $fold }
            if (Test-Path (Result-Path $job)) { continue }
            if ($runningNow["$($exp):$($seed):$($fold)"]) { continue }
            $jobs += $job
        }
    }
}

# fold1/fold2 are still training; do not start another job on those GPUs.
$external = @{
    1 = @{ Pid = 14520; Job = [pscustomobject]@{ Exp = "E1"; Seed = 43; Fold = 1 } }
    2 = @{ Pid = 15496; Job = [pscustomobject]@{ Exp = "E1"; Seed = 43; Fold = 2 } }
}
$running = @{}
$next = 0
$done = @(Get-ChildItem results\p0_legacy_D0 -Recurse -Filter result.json | Where-Object { $_.FullName -match "seed4[34]" }).Count
$failed = @()
$total = 20

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

function Finish-Job($job, $proc) {
    $tag = "{0}_s{1}_fold{2}" -f $job.Exp, $job.Seed, $job.Fold
    $ok = Test-Path (Result-Path $job)
    if (-not $ok -and $null -ne $proc -and $proc.ExitCode -ne 0) { $ok = $false }
    if ($ok) {
        $script:done++
        Write-Host "DONE $tag ($script:done/$total) $(Get-Date -Format o)"
        return
    }
    $code = if ($null -eq $proc) { "missing-result" } else { $proc.ExitCode }
    Write-Host "FAILED $tag exit $code"
    $script:failed += $tag
}

Write-Host "continue runner started $(Get-Date -Format o) queued=$($jobs.Count) already_done=$done"
while ($true) {
    foreach ($gpu in @($external.Keys)) {
        $item = $external[$gpu]
        $alive = Get-Process -Id $item.Pid -ErrorAction SilentlyContinue
        if ($alive) { continue }
        Finish-Job $item.Job $null
        $external.Remove($gpu)
    }

    $finished = @()
    foreach ($gpu in @($running.Keys)) {
        $item = $running[$gpu]
        $item.Proc.Refresh()
        if (-not $item.Proc.HasExited) { continue }
        try { $item.Proc.WaitForExit() } catch {}
        Finish-Job $item.Job $item.Proc
        $finished += $gpu
    }
    foreach ($gpu in $finished) { $running.Remove($gpu) }

    if ($failed.Count -eq 0) {
        foreach ($gpu in @(0, 1, 2)) {
            if ($running.ContainsKey($gpu) -or $external.ContainsKey($gpu)) { continue }
            if ($next -ge $jobs.Count) { continue }
            $job = $jobs[$next]
            if (Test-Path (Result-Path $job)) { $next++; continue }
            $next++
            $running[$gpu] = @{ Proc = (Start-FoldJob $job $gpu); Job = $job }
        }
    }

    if ($running.Count -eq 0 -and $external.Count -eq 0 -and ($next -ge $jobs.Count -or $failed.Count -gt 0)) { break }
    Start-Sleep -Seconds 20
}

if ($failed.Count -gt 0) {
    Write-Host "FAILED tasks: $($failed -join ', ')"
    exit 1
}

Write-Host "二十次训练完成 $(Get-Date -Format o)"
& $py summarize_ablations.py results/p0_legacy_D0 --output results/p0_legacy_D0_summary_seeds42_43_44
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "continue runner finished $(Get-Date -Format o)"
