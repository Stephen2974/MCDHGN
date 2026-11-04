$ErrorActionPreference = 'Stop'
try {
  $env:DGLBACKEND = 'pytorch'
  $env:PYTHONUNBUFFERED = '1'
  $env:CUDA_VISIBLE_DEVICES = '0,1,2'
  Set-Location -LiteralPath 'G:\LLLM\MCDHGN-main'
  $proc = Start-Process -FilePath 'G:\LLLM\MCDHGN-main\.conda\python.exe' -ArgumentList @('G:\LLLM\MCDHGN-main\results\lr003_E1_E3_seed42_20261104_111249\scheduler.py', 'G:\LLLM\MCDHGN-main\results\lr003_E1_E3_seed42_20261104_111249') -WorkingDirectory 'G:\LLLM\MCDHGN-main' -RedirectStandardOutput 'G:\LLLM\MCDHGN-main\results\logs\lr003_E1_E3_seed42_20261104_111249.out.log' -RedirectStandardError 'G:\LLLM\MCDHGN-main\results\logs\lr003_E1_E3_seed42_20261104_111249.err.log' -WindowStyle Hidden -PassThru
  @{
    scheduler_pid = $proc.Id
    python = 'G:\LLLM\MCDHGN-main\.conda\python.exe'
    output = 'G:\LLLM\MCDHGN-main\results\lr003_E1_E3_seed42_20261104_111249'
    scheduler = 'G:\LLLM\MCDHGN-main\results\lr003_E1_E3_seed42_20261104_111249\scheduler.py'
    scheduler_sha256 = '193eebd9a7ea429d24080a693af249f5d2815297e5eb10d13631ce65c81ac0dc'
    compare_sha256 = '238b9f70584a5337848fcbd7497c22d8c91925f42e8cb7909903c7bbe2943f7f'
    stdout = 'G:\LLLM\MCDHGN-main\results\logs\lr003_E1_E3_seed42_20261104_111249.out.log'
    stderr = 'G:\LLLM\MCDHGN-main\results\logs\lr003_E1_E3_seed42_20261104_111249.err.log'
    git_head = '7954bbc75186a74f23380efad6ef9fb4c1a13422'
    started = (Get-Date).ToString('o')
    DGLBACKEND = $env:DGLBACKEND
    PYTHONUNBUFFERED = $env:PYTHONUNBUFFERED
    CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
  } | ConvertTo-Json | Set-Content -LiteralPath 'G:\LLLM\MCDHGN-main\results\logs\lr003_E1_E3_seed42_20261104_111249.handoff.json' -Encoding UTF8
} catch {
  $_ | Out-File -LiteralPath 'G:\LLLM\MCDHGN-main\results\logs\lr003_E1_E3_seed42_20261104_111249.start.err' -Encoding utf8
  throw
}
