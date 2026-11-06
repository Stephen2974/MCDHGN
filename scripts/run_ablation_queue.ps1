# Free-GPU fold queue for run_ablation.py.
# As soon as a GPU finishes a fold, the next pending fold starts on that GPU.
# Do not wait for a whole wave to finish before filling idle cards.
#
# Example:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_ablation_queue.ps1 `
#     -Experiments E2 -Seeds 42 -Folds 0,1,2,3,4 -Devices 0,1,2 `
#     -DataRoot data_variants\d1_cnv_unverified_v1 `
#     -Output results\d1_unverified_E2_seed42_YYYYMMDD `
#     -SummarizeOutput results\d1_unverified_E2_seed42_YYYYMMDD_summary

[CmdletBinding()]
param(
    [string[]]$Experiments = @("E2"),
    [int[]]$Seeds = @(42),
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int[]]$Devices = @(0, 1, 2),
    [string]$DataRoot = ".",
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$SummarizeOutput = "",
    [string]$Python = "G:\LLLM\MCDHGN-main\.conda\python.exe",
    [string]$ProjectRoot = "G:\LLLM\MCDHGN-main",
    [int]$PollSeconds = 20,
    [int]$Epochs = 1000,
    [int]$EvalEvery = 10,
    [int]$Patience = 20,
    [double]$Lr = 0.01,
    [double]$WeightDecay = 0.0001,
    [double]$PositiveWeight = 1.904,
    [double]$Dropout = 0.4,
    [int]$HiddenSize = 256,
    [int]$Heads = 4,
    [int]$FusionDim = 64,
    [int]$FusionChunkSize = 1024,
    [switch]$SkipSummarize
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
$env:DGLBACKEND = "pytorch"
$env:PYTHONUNBUFFERED = "1"

$Output = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Output))
$DataRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $DataRoot))
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$runnerLog = Join-Path $Output "runner.log"

function Note([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $runnerLog -Value $line -Encoding utf8
    Write-Host $line
}

$jobs = New-Object System.Collections.Generic.List[object]
foreach ($exp in $Experiments) {
    foreach ($seed in $Seeds) {
        foreach ($fold in $Folds) {
            $jobs.Add([pscustomobject]@{ Exp = $exp; Seed = $seed; Fold = $fold })
        }
    }
}

$gpus = @($Devices)
$running = @{}
$next = 0
$done = 0
$failed = New-Object System.Collections.Generic.List[string]

function Start-FoldJob($job, [int]$Gpu) {
    $tag = "{0}_s{1}_fold{2}" -f $job.Exp, $job.Seed, $job.Fold
    $proc = Start-Process -FilePath $Python -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Output "$tag.log") `
        -RedirectStandardError (Join-Path $Output "$tag.err") `
        -ArgumentList @(
            "run_ablation.py",
            "--experiments", $job.Exp,
            "--seeds", "$($job.Seed)",
            "--folds", "$($job.Fold)",
            "--data-root", $DataRoot,
            "--devices", "$Gpu",
            "--output", $Output,
            "--epochs", "$Epochs",
            "--eval-every", "$EvalEvery",
            "--patience", "$Patience",
            "--lr", "$Lr",
            "--weight-decay", "$WeightDecay",
            "--positive-weight", "$PositiveWeight",
            "--dropout", "$Dropout",
            "--hidden-size", "$HiddenSize",
            "--heads", "$Heads",
            "--fusion-dim", "$FusionDim",
            "--fusion-chunk-size", "$FusionChunkSize"
        )
    Note "START $tag -> GPU $Gpu pid $($proc.Id)"
    return $proc
}

Note ("queue start experiments={0} seeds={1} folds={2} devices={3} jobs={4}" -f `
    ($Experiments -join ","), ($Seeds -join ","), ($Folds -join ","), ($Devices -join ","), $jobs.Count)

while ($true) {
    $finished = @()
    foreach ($gpu in @($running.Keys)) {
        $item = $running[$gpu]
        $item.Proc.Refresh()
        if (-not $item.Proc.HasExited) { continue }
        # Start-Process can leave ExitCode $null briefly; WaitForExit materializes it.
        # In PowerShell, ($null -ne 0) is $true, which falsely marks successful jobs failed.
        if ($null -eq $item.Proc.ExitCode) {
            $null = $item.Proc.WaitForExit(5000)
            $item.Proc.Refresh()
        }
        $tag = "{0}_s{1}_fold{2}" -f $item.Job.Exp, $item.Job.Seed, $item.Job.Fold
        $code = $item.Proc.ExitCode
        $resultPath = Join-Path $Output ("{0}\seed{1}\fold{2}\result.json" -f $item.Job.Exp, $item.Job.Seed, $item.Job.Fold)
        if (($null -eq $code -or $code -ne 0) -and -not (Test-Path -LiteralPath $resultPath)) {
            Note "FAILED $tag exit $code; see $tag.err"
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
    Start-Sleep -Seconds $PollSeconds
}

if ($failed.Count -gt 0) {
    Note ("FAILED tasks: " + ($failed -join ", "))
    exit 1
}

if (-not $SkipSummarize) {
    if (-not $SummarizeOutput) { $SummarizeOutput = "$Output`_summary" }
    $SummarizeOutput = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $SummarizeOutput))
    Note "summarize -> $SummarizeOutput"
    & $Python summarize_ablations.py $Output --output $SummarizeOutput
    if ($LASTEXITCODE -ne 0) {
        Note "summarize exit $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Note "queue finished"
