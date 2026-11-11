$ErrorActionPreference = "Continue"
$out = "G:\LLLM\MCDHGN-main\results\D1_unverified_lr003_E2_E4_seed42_20261108_212833"
$supervisor = 364
$log = Join-Path $out "watch.log"
while ($true) {
    $results = @(Get-ChildItem -LiteralPath $out -Recurse -Filter result.json -ErrorAction SilentlyContinue)
    $alive = [bool](Get-Process -Id $supervisor -ErrorAction SilentlyContinue)
    $py = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -and $_.CommandLine -like "*D1_unverified_lr003_E2_E4_seed42_20261108_212833*"
    })
    $gpu = (nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader) -join " | "
    $errHits = @(Get-ChildItem -LiteralPath $out -Filter "*.err" | Where-Object { $_.Length -gt 0 -and $_.Name -notlike "queue_console*" })
    $line = "{0} results={1}/10 python={2} supervisor={3} nonempty_err={4} gpu=[{5}]" -f (Get-Date -Format "HH:mm:ss"), $results.Count, $py.Count, $alive, $errHits.Count, $gpu
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
    Write-Output $line
    if ($results.Count -ge 10 -and $py.Count -eq 0) {
        Write-Output "ALL_DONE"
        break
    }
    if (-not $alive -and $py.Count -eq 0) {
        Write-Output ("SUPERVISOR_STOPPED results=" + $results.Count)
        break
    }
    Start-Sleep -Seconds 120
}
