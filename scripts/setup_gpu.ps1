<#
.SYNOPSIS
  Install a GPU-enabled PyTorch build into this project's .venv.

.DESCRIPTION
  The tracked uv.lock is hardware-neutral. On Windows, PyPI's torch wheel is
  CPU-only, so run this script after `uv sync` (or after changing dependencies)
  to install the CUDA build that matches your driver.

  Requires uv (on PATH, or available through `py -3.13 -m uv`).

  Note: `uv sync` / `uv run` will replace this CUDA build with the locked CPU
  wheel. Use `uv run --no-sync ...` (or set UV_NO_SYNC=1) for day-to-day work,
  and rerun this script after any sync.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1

.EXAMPLE
  # Faster PyTorch mirror (e.g. in China)
  powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1 -IndexUrl https://mirror.sjtu.edu.cn/pytorch-wheels/cu130
#>
[CmdletBinding()]
param(
    [switch]$SkipSync,
    [switch]$VerifyOnly,
    [string]$IndexUrl = '',
    [string]$TorchBackend = 'auto'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        function Invoke-Uv { & uv @args }
    } else {
        function Invoke-Uv { & py -3.13 -m uv @args }
    }

    if (-not $VerifyOnly) {
        if (-not $SkipSync) {
            Write-Host '== uv sync =='
            Invoke-Uv sync
        }
        Write-Host '== install GPU torch =='
        if ($IndexUrl) {
            Invoke-Uv pip install --reinstall torch torchvision --index-url $IndexUrl
        } else {
            Invoke-Uv pip install --reinstall torch torchvision "--torch-backend=$TorchBackend"
        }
    }

    Write-Host '== verify =='
    Invoke-Uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
} finally {
    Pop-Location
}
