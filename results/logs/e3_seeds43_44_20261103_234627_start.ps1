$ErrorActionPreference = 'Stop'
try {
  $env:DGLBACKEND = 'pytorch'
  $env:PYTHONUNBUFFERED = '1'
  $env:CUDA_VISIBLE_DEVICES = '0,1,2'
  Set-Location -LiteralPath 'G:\LLLM\MCDHGN-main'
  $proc = Start-Process -FilePath 'G:\LLLM\MCDHGN-main\.conda\python.exe' -ArgumentList @('G:\LLLM\MCDHGN-main\results\d0_E3_seeds43_44_20261103_234627\scheduler.py', 'G:\LLLM\MCDHGN-main\results\d0_E3_seeds43_44_20261103_234627') -WorkingDirectory 'G:\LLLM\MCDHGN-main' -RedirectStandardOutput 'G:\LLLM\MCDHGN-main\results\logs\e3_seeds43_44_20261103_234627.out.log' -RedirectStandardError 'G:\LLLM\MCDHGN-main\results\logs\e3_seeds43_44_20261103_234627.err.log' -WindowStyle Hidden -PassThru
  @{
    scheduler_pid = $proc.Id
    python = 'G:\LLLM\MCDHGN-main\.conda\python.exe'
    output = 'G:\LLLM\MCDHGN-main\results\d0_E3_seeds43_44_20261103_234627'
    scheduler = 'G:\LLLM\MCDHGN-main\results\d0_E3_seeds43_44_20261103_234627\scheduler.py'
    scheduler_sha256 = '11c312e8b0e33a1d415ed9ddfeabf45f0d0d593c81b4fbfab12060a104217332'
    stdout = 'G:\LLLM\MCDHGN-main\results\logs\e3_seeds43_44_20261103_234627.out.log'
    stderr = 'G:\LLLM\MCDHGN-main\results\logs\e3_seeds43_44_20261103_234627.err.log'
    git_head = '7954bbc75186a74f23380efad6ef9fb4c1a13422'
    started = (Get-Date).ToString('o')
    DGLBACKEND = $env:DGLBACKEND
    PYTHONUNBUFFERED = $env:PYTHONUNBUFFERED
    CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
    command = 'G:\LLLM\MCDHGN-main\.conda\python.exe G:\LLLM\MCDHGN-main\results\d0_E3_seeds43_44_20261103_234627\scheduler.py G:\LLLM\MCDHGN-main\results\d0_E3_seeds43_44_20261103_234627'
  } | ConvertTo-Json | Set-Content -LiteralPath 'G:\LLLM\MCDHGN-main\results\logs\e3_seeds43_44_20261103_234627.handoff.json' -Encoding UTF8
} catch {
  $_ | Out-File -LiteralPath 'G:\LLLM\MCDHGN-main\results\logs\e3_seeds43_44_20261103_234627.start.err' -Encoding utf8
  throw
}
