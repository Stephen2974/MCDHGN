# One-off free-GPU queue for this lr=0.003 comparison only.
# Priority: E2 fold4, E4 fold4, E4 fold0, then the remaining seven folds.
# One fold per GPU. Do not stop foreign processes. Do not retry failures.
$ErrorActionPreference = "Stop"
$ProjectRoot = "G:\LLLM\MCDHGN-main"
$Python = "G:\LLLM\MCDHGN-main\.conda\python.exe"
$DataRoot = "G:\LLLM\MCDHGN-main\data_variants\d1_cnv_unverified_v1"
$Output = "G:\LLLM\MCDHGN-main\results\D1_unverified_lr003_E2_E4_seed42_20261108_212833"
$Devices = @(0, 1, 2)
$PollSeconds = 20
$MemoryBusyMiB = 2048

Set-Location -LiteralPath $ProjectRoot
$env:DGLBACKEND = "pytorch"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$runnerLog = Join-Path $Output "runner.log"
$commandLog = Join-Path $Output "commands.log"

function Note([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $runnerLog -Value $line -Encoding utf8
    Write-Host $line
}

$jobs = @(
    [pscustomobject]@{ Exp = "E2"; Seed = 42; Fold = 4 },
    [pscustomobject]@{ Exp = "E4"; Seed = 42; Fold = 4 },
    [pscustomobject]@{ Exp = "E4"; Seed = 42; Fold = 0 },
    [pscustomobject]@{ Exp = "E2"; Seed = 42; Fold = 0 },
    [pscustomobject]@{ Exp = "E2"; Seed = 42; Fold = 1 },
    [pscustomobject]@{ Exp = "E2"; Seed = 42; Fold = 2 },
    [pscustomobject]@{ Exp = "E2"; Seed = 42; Fold = 3 },
    [pscustomobject]@{ Exp = "E4"; Seed = 42; Fold = 1 },
    [pscustomobject]@{ Exp = "E4"; Seed = 42; Fold = 2 },
    [pscustomobject]@{ Exp = "E4"; Seed = 42; Fold = 3 }
)

function Get-BlockedGpus {
    # Emit GPU indexes one per line. An empty result must stay an empty array
    # at the call site; do not return a HashSet (PowerShell enumerates it).
    $blocked = New-Object System.Collections.Generic.List[int]
    $uuidToIndex = @{}
    $gpuRows = @(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0) {
        Note "WAIT nvidia-smi gpu query failed; not starting folds this cycle"
        foreach ($index in $Devices) { Write-Output $index }
        return
    }
    foreach ($row in $gpuRows) {
        $parts = @($row.Split(",") | ForEach-Object { $_.Trim() })
        $index = [int]$parts[0]
        $uuidToIndex[$parts[1]] = $index
        $used = [int]($parts[2] -replace "[^0-9]", "")
        if ($used -ge $MemoryBusyMiB) { [void]$blocked.Add($index) }
    }
    $apps = @(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader)
    foreach ($row in $apps) {
        if (-not $row -or $row -notmatch ",") { continue }
        $parts = @($row.Split(",") | ForEach-Object { $_.Trim() })
        $name = $parts[$parts.Length - 1]
        if ($name -match '(?i)python(\.exe)?$') {
            $uuid = $parts[0]
            if ($uuidToIndex.ContainsKey($uuid)) { [void]$blocked.Add($uuidToIndex[$uuid]) }
        }
    }
    $blocked | Select-Object -Unique | ForEach-Object { Write-Output $_ }
}

function Start-FoldJob($job, [int]$Gpu) {
    $tag = "{0}_s{1}_fold{2}" -f $job.Exp, $job.Seed, $job.Fold
    $argList = @(
        "run_ablation.py",
        "--experiments", $job.Exp,
        "--seeds", "$($job.Seed)",
        "--folds", "$($job.Fold)",
        "--data-root", $DataRoot,
        "--devices", "$Gpu",
        "--output", $Output,
        "--epochs", "1000",
        "--eval-every", "10",
        "--patience", "20",
        "--lr", "0.003",
        "--weight-decay", "0.0001",
        "--positive-weight", "1.904",
        "--dropout", "0.4",
        "--hidden-size", "256",
        "--heads", "4",
        "--fusion-dim", "64",
        "--fusion-chunk-size", "1024"
    )
    $cmd = "$Python " + ($argList -join " ")
    Add-Content -LiteralPath $commandLog -Value "$(Get-Date -Format o) GPU $Gpu $cmd" -Encoding utf8
    $proc = Start-Process -FilePath $Python -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Output "$tag.log") `
        -RedirectStandardError (Join-Path $Output "$tag.err") `
        -ArgumentList $argList
    Note "START $tag -> GPU $Gpu pid $($proc.Id)"
    Note "CMD $cmd"
    return $proc
}

Note "queue start lr=0.003 experiments=E2,E4 seed=42 folds=priority-then-rest devices=0,1,2 jobs=$($jobs.Count) data=$DataRoot"
Note "git HEAD 37b58186db2f3729aed4714f96048bb975828e62; uncommitted work left untouched; no test-run"

$running = @{}
$next = 0
$done = 0
$failed = New-Object System.Collections.Generic.List[string]

while ($true) {
    $finished = @()
    foreach ($gpu in @($running.Keys)) {
        $item = $running[$gpu]
        $item.Proc.Refresh()
        if (-not $item.Proc.HasExited) { continue }
        if ($null -eq $item.Proc.ExitCode) {
            $null = $item.Proc.WaitForExit(5000)
            $item.Proc.Refresh()
        }
        $tag = "{0}_s{1}_fold{2}" -f $item.Job.Exp, $item.Job.Seed, $item.Job.Fold
        $code = $item.Proc.ExitCode
        $resultPath = Join-Path $Output ("{0}\seed{1}\fold{2}\result.json" -f $item.Job.Exp, $item.Job.Seed, $item.Job.Fold)
        if (($null -eq $code -or $code -ne 0) -and -not (Test-Path -LiteralPath $resultPath)) {
            Note "FAILED $tag exit $code; see $tag.err; not retrying"
            $failed.Add($tag) | Out-Null
        } else {
            if ($null -eq $code -or $code -ne 0) {
                Note "WARN $tag exit=$code but result.json present; treating as DONE"
            }
            $done++
            Note "DONE $tag ($done/$($jobs.Count))"
        }
        $finished += $gpu
    }
    foreach ($gpu in $finished) { $running.Remove($gpu) }

    $blocked = @(Get-BlockedGpus)
    foreach ($gpu in $Devices) {
        if ($running.ContainsKey($gpu)) { continue }
        if ($next -ge $jobs.Count) { continue }
        if ($blocked -contains $gpu) {
            Note "WAIT GPU $gpu busy with other compute or memory>=$MemoryBusyMiB; not interrupting"
            continue
        }
        $job = $jobs[$next]
        $next++
        $running[$gpu] = @{ Proc = (Start-FoldJob $job $gpu); Job = $job }
    }

    if ($running.Count -eq 0 -and $next -ge $jobs.Count) { break }
    Start-Sleep -Seconds $PollSeconds
}

if ($failed.Count -gt 0) {
    Note ("FAILED tasks retained: " + ($failed -join ", "))
    exit 1
}
Note "queue finished"
exit 0
