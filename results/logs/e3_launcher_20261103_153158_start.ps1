$ErrorActionPreference = 'Stop'
try {
  $env:DGLBACKEND = 'pytorch'
  $env:PYTHONUNBUFFERED = '1'
  $env:CUDA_VISIBLE_DEVICES = '0,1,2'
  Set-Location -LiteralPath 'G:\LLLM\MCDHGN-main'
  $proc = Start-Process -FilePath 'G:\LLLM\MCDHGN-main\.conda\python.exe' -ArgumentList @('scripts/run_e3_parallel.py', '--output', 'G:\LLLM\MCDHGN-main\results\d0_E3_seed42_20261103_153158') -WorkingDirectory 'G:\LLLM\MCDHGN-main' -RedirectStandardOutput 'G:\LLLM\MCDHGN-main\results\logs\e3_launcher_20261103_153158.out.log' -RedirectStandardError 'G:\LLLM\MCDHGN-main\results\logs\e3_launcher_20261103_153158.err.log' -WindowStyle Hidden -PassThru
  @{
    launcher_pid = $proc.Id
    python = 'G:\LLLM\MCDHGN-main\.conda\python.exe'
    output = 'G:\LLLM\MCDHGN-main\results\d0_E3_seed42_20261103_153158'
    stdout = 'G:\LLLM\MCDHGN-main\results\logs\e3_launcher_20261103_153158.out.log'
    stderr = 'G:\LLLM\MCDHGN-main\results\logs\e3_launcher_20261103_153158.err.log'
    started = (Get-Date).ToString('o')
    DGLBACKEND = $env:DGLBACKEND
    PYTHONUNBUFFERED = $env:PYTHONUNBUFFERED
    CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
  } | ConvertTo-Json | Set-Content -LiteralPath 'G:\LLLM\MCDHGN-main\results\logs\e3_launcher_20261103_153158.meta.json' -Encoding UTF8
} catch {
  $_ | Out-File -LiteralPath 'G:\LLLM\MCDHGN-main\results\logs\e3_launcher_20261103_153158.start.err' -Encoding utf8
  throw
}
