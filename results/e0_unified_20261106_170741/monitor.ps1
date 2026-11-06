# Real-time E0 training + download monitor. Ctrl+C to stop.
# Force UTF-8 so Chinese logs are not mojibake in Cursor terminals.
$ErrorActionPreference = "Continue"
try {
    chcp 65001 | Out-Null
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}

$out = "G:\LLLM\MCDHGN-main\results\e0_unified_20261106_170741"
$d2src = "G:\LLLM\MCDHGN-main\data_versions\d2_mutation_v1\sources"

function Read-Utf8Tail([string]$Path, [int]$N = 2) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    Get-Content -LiteralPath $Path -Tail $N -Encoding UTF8
}

while ($true) {
    Clear-Host
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "==== $stamp  E0 monitor ====" -ForegroundColor Cyan

    $done = @(); $pending = @()
    foreach ($seed in 42, 43, 44) {
        foreach ($fold in 0..4) {
            $tag = "s$seed-f$fold"
            if (Test-Path (Join-Path $out "E0\seed$seed\fold$fold\result.json")) { $done += $tag } else { $pending += $tag }
        }
    }
    Write-Host ("E0 complete: {0}/15" -f $done.Count) -ForegroundColor Green
    if ($pending.Count) { Write-Host ("Pending: {0}" -f ($pending -join ", ")) -ForegroundColor Yellow }

    Write-Host ""
    Write-Host "-- runner.log (tail) --"
    Read-Utf8Tail (Join-Path $out "runner.log") 8 | ForEach-Object { Write-Host $_ }

    Write-Host ""
    Write-Host "-- active fold logs (tail) --"
    Get-ChildItem (Join-Path $out "E0_s*.log") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 3 |
        ForEach-Object {
            Write-Host ("[{0}] {1}" -f $_.LastWriteTime.ToString("HH:mm:ss"), $_.Name) -ForegroundColor DarkCyan
            Read-Utf8Tail $_.FullName 2 | ForEach-Object { Write-Host $_ }
        }

    Write-Host ""
    Write-Host "-- GPU --"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader

    Write-Host ""
    Write-Host "-- python jobs --"
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
        $c = $_.CommandLine
        if (-not $c) { return }
        if ($c -match 'run_ablation') {
            if ($c -match '--seeds (\d+).*--folds (\d+).*--devices (\d+)') {
                Write-Host ("train  PID={0} seed={1} fold={2} gpu={3}" -f $_.ProcessId, $Matches[1], $Matches[2], $Matches[3])
            } else {
                Write-Host ("train  PID={0}" -f $_.ProcessId)
            }
        } elseif ($c -match 'data_processing\.(d1|d2)') {
            Write-Host ("download PID={0}" -f $_.ProcessId)
            Write-Host ("  {0}" -f $c)
        }
    }

    Write-Host ""
    Write-Host "-- D2 download progress (ASCII) --"
    if (Test-Path -LiteralPath $d2src) {
        $studies = Get-ChildItem -LiteralPath $d2src -Directory
        $nDone = 0; $nPartial = 0; $nMissing = 0
        foreach ($st in $studies) {
            $mut = Join-Path $st.FullName "data_mutations.txt"
            $part = Join-Path $st.FullName "data_mutations.txt.partial"
            if (Test-Path -LiteralPath $mut) { $nDone++ }
            elseif (Test-Path -LiteralPath $part) {
                $nPartial++
                $len = (Get-Item -LiteralPath $part).Length
                Write-Host ("  PARTIAL {0}  {1:N1} MB  mtime={2}" -f $st.Name, ($len / 1MB), (Get-Item $part).LastWriteTime.ToString("HH:mm:ss"))
            } else { $nMissing++ }
        }
        Write-Host ("  studies seen={0}  mutations done={1}  partial={2}  not_started={3}" -f $studies.Count, $nDone, $nPartial, $nMissing)
    } else {
        Write-Host "  sources dir missing"
    }

    Write-Host ""
    Write-Host "-- download events (time/step/status only) --"
    foreach ($ev in @(
        "G:\LLLM\MCDHGN-main\data_versions\d2_mutation_v1\events.jsonl",
        "G:\LLLM\MCDHGN-main\data_versions\d1_cnv_v1\events.jsonl"
    )) {
        if (-not (Test-Path -LiteralPath $ev)) { continue }
        Write-Host ("  file={0}" -f $ev)
        foreach ($line in (Read-Utf8Tail $ev 2)) {
            try {
                $obj = $line | ConvertFrom-Json
                $step = [string]$obj.step
                # Keep ASCII-safe: show raw step string; avoid dumping Chinese details blob.
                Write-Host ("  {0} | step={1} | status={2}" -f $obj.time, $step, $obj.status)
            } catch {
                # Fallback: strip non-ASCII so terminal never shows mojibake blocks.
                $safe = -join ($line.ToCharArray() | Where-Object { [int]$_ -lt 128 })
                Write-Host ("  {0}" -f $safe)
            }
        }
    }

    Start-Sleep -Seconds 15
}
