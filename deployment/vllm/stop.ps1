$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if ($null -eq $wsl) {
    throw "wsl.exe was not found. Enable WSL2 and install an Ubuntu distribution."
}

& $wsl.Source --cd $repoRoot -e bash ./deployment/vllm/stop.sh
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
