# Read-only live view of this D1 run. Does not start, stop, or change training.
param(
    [string]$Output = "G:\LLLM\MCDHGN-main\results\D1_unverified_remaining_E0_E1_E3_E5_E6_E7_lr001_seed42_20261109_151202",
    [int]$RefreshSeconds = 10,
    [int]$MaxEpochs = 1000,
    [int]$PatienceChecks = 20,
    [int]$EvalEvery = 10,
    [int]$Parallel = 3,
    [int]$TotalJobs = 30,
    [int]$ReferenceEpochs = 775,
    [switch]$Once
)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try { $Host.UI.RawUI.WindowTitle = "D1 AP/AUC monitor" } catch {}
$Output = (Resolve-Path -LiteralPath $Output).Path

function Format-Span([double]$Seconds) {
    if ($Seconds -lt 0 -or [double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds)) { return "-" }
    $span = [TimeSpan]::FromSeconds([Math]::Max(0, $Seconds))
    if ($span.TotalHours -ge 1) { return "{0:0}h{1:00}m" -f [Math]::Floor($span.TotalHours), $span.Minutes }
    if ($span.TotalMinutes -ge 1) { return "{0:0}m{1:00}s" -f [Math]::Floor($span.TotalMinutes), $span.Seconds }
    return "{0:0}s" -f $span.TotalSeconds
}

function Get-Runs {
    $runner = Join-Path $Output "runner.log"
    $assign = @{}
    if (Test-Path -LiteralPath $runner) {
        foreach ($line in @(Get-Content -LiteralPath $runner -Encoding UTF8 -ErrorAction SilentlyContinue)) {
            if ($line -match "START (E\d)_s(\d+)_fold(\d+) -> GPU (\d+) pid (\d+)") {
                $assign["$($Matches[1])|$($Matches[2])|$($Matches[3])"] = [pscustomobject]@{
                    Gpu = [int]$Matches[4]; Pid = [int]$Matches[5]; Started = $line.Substring(0, 33)
                }
            }
        }
    }

    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($log in @(Get-ChildItem -LiteralPath $Output -Filter "*_s*_fold*.log" -File -ErrorAction SilentlyContinue)) {
        if ($log.Name -notmatch "^(E\d)_s(\d+)_fold(\d+)\.log$") { continue }
        $exp, $seed, $fold = $Matches[1], [int]$Matches[2], [int]$Matches[3]
        $evals = New-Object System.Collections.Generic.List[object]
        foreach ($line in @(Get-Content -LiteralPath $log.FullName -Encoding UTF8 -ErrorAction SilentlyContinue)) {
            if ($line -match "epoch=(\d+) AP=([0-9.eE+-]+) AUC=([0-9.eE+-]+)") {
                $evals.Add([pscustomobject]@{
                    Epoch = [int]$Matches[1]; Ap = [double]$Matches[2]; Auc = [double]$Matches[3]
                }) | Out-Null
            }
        }
        $best = $null
        foreach ($ev in $evals) {
            if ($null -eq $best -or $ev.Ap -gt $best.Ap) { $best = $ev }
        }
        $last = if ($evals.Count) { $evals[$evals.Count - 1] } else { $null }
        $key = "$exp|$seed|$fold"
        $meta = $assign[$key]
        $resultPath = Join-Path $Output "$exp\seed$seed\fold$fold\result.json"
        $done = Test-Path -LiteralPath $resultPath
        $result = $null
        if ($done) {
            try { $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $result = $null }
        }
        $alive = $false
        if ($meta -and -not $done) {
            $alive = $null -ne (Get-Process -Id $meta.Pid -ErrorAction SilentlyContinue)
        }
        $startedAt = $null
        if ($meta -and $meta.Started) {
            try { $startedAt = [datetime]::Parse($meta.Started) } catch { $startedAt = $null }
        }
        $rows.Add([pscustomobject]@{
            Exp = $exp; Seed = $seed; Fold = $fold; Gpu = $(if ($meta) { $meta.Gpu } else { $null })
            Pid = $(if ($meta) { $meta.Pid } else { $null }); StartedAt = $startedAt
            Last = $last; Best = $best; Evals = $evals.Count; Log = $log
            Done = $done; Alive = $alive; Result = $result
        }) | Out-Null
    }
    return $rows
}

while ($true) {
    $now = Get-Date
    $runs = @(Get-Runs)
    $rates = New-Object System.Collections.Generic.List[double]
    $view = New-Object System.Collections.Generic.List[object]
    foreach ($run in $runs) {
        $epoch = 0; $ap = $null; $auc = $null; $bestEpoch = $null; $bestAp = $null; $rate = $null
        $remainPatience = $null; $remainMax = $null; $state = "wait"
        if ($run.Last) {
            $epoch = $run.Last.Epoch; $ap = $run.Last.Ap; $auc = $run.Last.Auc
            $bestEpoch = $run.Best.Epoch; $bestAp = $run.Best.Ap
        }
        if ($run.Done -and $run.Result) {
            $state = "done"
            $epoch = [int]$run.Result.epochs_trained
            $bestEpoch = [int]$run.Result.best_epoch
            $ap = [double]$run.Result.validation.ap
            $auc = [double]$run.Result.validation.auc
            $bestAp = $ap
            if ($run.Result.train_seconds -and $epoch -gt 0) { $rate = [double]$run.Result.train_seconds / $epoch }
        } elseif ($run.Alive) {
            $state = "run"
            if ($run.StartedAt -and $epoch -gt 0) {
                $rate = ($now - $run.StartedAt).TotalSeconds / $epoch
                $rates.Add($rate) | Out-Null
                $stopEpoch = [Math]::Min($MaxEpochs, $bestEpoch + ($PatienceChecks * $EvalEvery))
                $remainPatience = [Math]::Max(0, $stopEpoch - $epoch) * $rate
                $remainMax = [Math]::Max(0, $MaxEpochs - $epoch) * $rate
            }
        } elseif ($run.Last -and -not $run.Done) {
            $state = "exit"
        }
        $view.Add([pscustomobject]@{
            State = $state; Exp = $run.Exp; Fold = $run.Fold; Gpu = $run.Gpu; Epoch = $epoch
            BestEpoch = $bestEpoch; Ap = $ap; Auc = $auc; BestAp = $bestAp; Rate = $rate
            PatienceLeft = $remainPatience; MaxLeft = $remainMax
        }) | Out-Null
    }

    $doneN = @($view | Where-Object State -eq "done").Count
    $runN = @($view | Where-Object State -eq "run").Count
    $exitN = @($view | Where-Object State -eq "exit").Count
    $seen = $view.Count
    $pending = [Math]::Max(0, $TotalJobs - $seen)
    $median = $null
    if ($rates.Count) {
        $sorted = @($rates | Sort-Object)
        $mid = [int][Math]::Floor(($sorted.Count - 1) / 2)
        $median = $sorted[$mid]
    }
    $runningLeftMax = ($view | Where-Object { $_.State -eq "run" -and $null -ne $_.MaxLeft } | Measure-Object MaxLeft -Sum).Sum
    $runningLeftPat = ($view | Where-Object { $_.State -eq "run" -and $null -ne $_.PatienceLeft } | Measure-Object PatienceLeft -Sum).Sum
    if ($null -eq $runningLeftMax) { $runningLeftMax = 0 }
    if ($null -eq $runningLeftPat) { $runningLeftPat = 0 }
    $queueLeftMax = $null; $queueLeftPat = $null; $queueLeftRef = $null
    $etaMax = $null; $etaPat = $null; $etaRef = $null
    if ($median) {
        $future = $pending
        $queueLeftMax = ($runningLeftMax + ($future * $MaxEpochs * $median)) / $Parallel
        $floorEpochs = [Math]::Min($MaxEpochs, $PatienceChecks * $EvalEvery)
        $queueLeftPat = ($runningLeftPat + ($future * $floorEpochs * $median)) / $Parallel
        $refEpochs = [Math]::Min($MaxEpochs, $ReferenceEpochs)
        $runningLeftRef = 0.0
        foreach ($row in $view) {
            if ($row.State -eq "run" -and $null -ne $row.Rate) {
                $runningLeftRef += [Math]::Max(0, $refEpochs - $row.Epoch) * $row.Rate
            }
        }
        $queueLeftRef = ($runningLeftRef + ($future * $refEpochs * $median)) / $Parallel
        $etaMax = $now.AddSeconds($queueLeftMax)
        $etaPat = $now.AddSeconds($queueLeftPat)
        $etaRef = $now.AddSeconds($queueLeftRef)
    }

    Clear-Host
    Write-Host ("D1 lr=0.01  E0/E1/E3/E5/E6/E7  seed42    {0:HH:mm:ss}" -f $now)
    Write-Host ("done {0}/{1}   running {2}   pending {3}   abnormal_exit {4}" -f $doneN, $TotalJobs, $runN, $pending, $exitN)
    if ($median) {
        Write-Host ("speed {0:0.00} s/epoch (includes data load; median of running folds)" -f $median)
        Write-Host ("lower bound, no further AP gain: about {0}, around {1:MM-dd HH:mm}" -f (Format-Span $queueLeftPat), $etaPat)
        Write-Host ("reference, 775 epochs like existing D1 E2/E4 lr=0.01: about {0}, around {1:MM-dd HH:mm}" -f (Format-Span $queueLeftRef), $etaRef)
        Write-Host ("upper bound, every remaining fold runs 1000 epochs: about {0}, around {1:MM-dd HH:mm}" -f (Format-Span $queueLeftMax), $etaMax)
    } else {
        Write-Host "speed -   waiting for the first epoch line"
    }
    Write-Host "patience=20 checks x 10 epochs. A later AP improvement pushes the early-stop point back."
    Write-Host ""
    "{0,-5} {1,-4} {2,-4} {3,6} {4,6} {5,8} {6,8} {7,8} {8,8} {9,8}" -f "state","exp","gpu","epoch","best","AP","AUC","bestAP","no+imp","to1000" | Write-Host
    foreach ($row in ($view | Sort-Object @{Expression = { if ($_.State -eq "run") { 0 } elseif ($_.State -eq "exit") { 1 } elseif ($_.State -eq "done") { 2 } else { 3 } }}, Exp, Fold)) {
        $gpu = if ($null -eq $row.Gpu) { "-" } else { "$($row.Gpu)" }
        $apS = if ($null -eq $row.Ap) { "-" } else { "{0:0.0000}" -f $row.Ap }
        $aucS = if ($null -eq $row.Auc) { "-" } else { "{0:0.0000}" -f $row.Auc }
        $bapS = if ($null -eq $row.BestAp) { "-" } else { "{0:0.0000}" -f $row.BestAp }
        $beS = if ($null -eq $row.BestEpoch) { "-" } else { "$($row.BestEpoch)" }
        "{0,-5} {1,-4} {2,-4} {3,6} {4,6} {5,8} {6,8} {7,8} {8,8} {9,8}" -f $row.State, $row.Exp, $gpu, $row.Epoch, $beS, $apS, $aucS, $bapS, (Format-Span $row.PatienceLeft), (Format-Span $row.MaxLeft) | Write-Host
    }
    Write-Host ""
    Write-Host "AP/AUC for a running fold are the latest validation point. For a done fold they are the saved best checkpoint."
    Write-Host "Ctrl+C exits this view only. Refresh ${RefreshSeconds}s."
    if ($Once) { break }
    Start-Sleep -Seconds $RefreshSeconds
}
