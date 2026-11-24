param(
    [Parameter(Mandatory=$true)]
    [string]$RemoteUrl
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .git)) {
    git init -b main
}
git add -A
git commit -m "Lab 6: Distributed Data-Parallel CNN on CPU (Variant 24, OCTMNIST)" --allow-empty
git remote remove origin 2>$null
git remote add origin $RemoteUrl
git branch -M main
git push -u origin main
