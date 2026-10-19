param(
    [string]$HostName = "unitree@192.168.1.3",
    [string]$Password = $(if ($env:IBVS_SSH_PASSWORD) { $env:IBVS_SSH_PASSWORD } else { "123" }),
    [string]$RemoteDir = "~/IBVS_PAD_TOUCH",
    [int]$Port = 8080,
    [switch]$Demo,
    [switch]$SkipUpload,
    [switch]$Tunnel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { $Python = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $Python) { throw "Need python on PATH to deploy over SSH." }

$Arguments = @(
    (Join-Path $ProjectRoot "scripts\h2_ssh.py"),
    "deploy",
    "--host", $HostName,
    "--password", $Password,
    "--remote-dir", $RemoteDir,
    "--port", "$Port"
)
if ($Demo) { $Arguments += "--demo" }
if ($SkipUpload) { $Arguments += "--skip-upload" }
if ($Tunnel) { $Arguments += "--tunnel" }

& $Python.Source @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "deploy_web failed with exit code $LASTEXITCODE"
}

$OpenUrl = if ($Tunnel) { "http://127.0.0.1:$Port/" } else { "http://192.168.1.3:$Port/" }
Start-Process $OpenUrl
Write-Host "Browser: $OpenUrl"
