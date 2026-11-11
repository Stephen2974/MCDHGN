# Real-time D1 unverified E4 seed42 AP/AUC monitor. Ctrl+C to stop.
$ErrorActionPreference = "Continue"
try {
    chcp 65001 | Out-Null
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}

$out = "G:\LLLM\MCDHGN-main\results\D1_unverified_E4_seed42_20261108_104318"

function Read-Utf8Tail([string]$Path, [int]$N = 3) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    return @(Get-Content -LiteralPath $Path -Tail $N -Encoding UTF8)
}

while ($true) {
    Clear-Host
    Write-Host ("==== {0}  D1 E4 seed42  AP / AUC ====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor Cyan
    Write-Host $out

    $done = 0
    Write-Host ""
    Write-Host ("{0,-6} {1,-10} {2,-12} {3,-12} {4}" -f "fold", "status", "AP", "AUC", "detail")
    foreach ($fold in 0..4) {
        $result = Join-Path $out "E4\seed42\fold$fold\result.json"
        $hist = Join-Path $out "E4\seed42\fold$fold\history.jsonl"
        if (Test-Path -LiteralPath $result) {
            $done++
            try {
                $j = Get-Content -LiteralPath $result -Raw -Encoding UTF8 | ConvertFrom-Json
                Write-Host ("{0,-6} {1,-10} {2,-12:N6} {3,-12:N6} best_epoch={4} trained={5}" -f "fold$fold", "DONE", $j.validation.ap, $j.validation.auc, $j.best_epoch, $j.epochs_trained) -ForegroundColor Green
            } catch {
                Write-Host "fold$fold DONE parse-error"
            }
        } elseif (Test-Path -LiteralPath $hist) {
            $last = @(Read-Utf8Tail $hist 1 | Where-Object { $_ -match '^\{' })
            if ($last.Count -ge 1) {
                try {
                    $row = $last[-1] | ConvertFrom-Json
                    Write-Host ("{0,-6} {1,-10} {2,-12:N6} {3,-12:N6} epoch={4}" -f "fold$fold", "running", $row.ap, $row.auc, $row.epoch)
                } catch {
                    Write-Host "fold$fold running metrics-updating"
                }
            } else {
                Write-Host ("{0,-6} {1,-10}" -f "fold$fold", "starting")
            }
        } else {
            Write-Host ("{0,-6} {1,-10}" -f "fold$fold", "queued") -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host ("complete {0}/5" -f $done) -ForegroundColor Green

    Write-Host ""
    Write-Host "-- latest log lines --"
    foreach ($fold in 0..4) {
        $log = Join-Path $out "E4_s42_fold$fold.log"
        $lines = @(Read-Utf8Tail $log 1 | Where-Object { $_ })
        if ($lines.Count -ge 1) { Write-Host ("fold{0}: {1}" -f $fold, $lines[-1]) }
    }

    Write-Host ""
    Write-Host "-- GPU --"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader

    Write-Host ""
    Write-Host "-- processes --"
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
        $c = $_.CommandLine
        if ($c -and $c -match 'D1_unverified_E4_seed42_20261108_104318' -and $c -match '--folds (\d+).*--devices (\d+)') {
            Write-Host ("PID={0} fold={1} gpu={2}" -f $_.ProcessId, $Matches[1], $Matches[2])
        }
    }

    Start-Sleep -Seconds 10
}
