# hipfire force-update — clean reinstall preserving models only.
# Config is regenerated fresh (ensures correct GPU arch detection).
# Usage: irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/force-update.ps1 | iex

$HipfireDir = "$env:USERPROFILE\.hipfire"
$BackupDir = "$env:USERPROFILE\.hipfire-backup"

Write-Host "=== hipfire force-update ===" -ForegroundColor Cyan

# Back up user data
if (Test-Path $HipfireDir) {
    Write-Host "Backing up models and config..."
    New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
    if (Test-Path "$HipfireDir\models")      { Copy-Item "$HipfireDir\models" "$BackupDir\models" -Recurse -Force; Write-Host "  models ✓" }
    # config.json intentionally NOT backed up — installer regenerates with fresh GPU detection
    if (Test-Path "$HipfireDir\runtime")     { Copy-Item "$HipfireDir\runtime" "$BackupDir\runtime" -Recurse -Force; Write-Host "  runtime (amdhip64.dll) ✓" }

    Write-Host "Removing old install..."
    Remove-Item -Recurse -Force $HipfireDir
    Write-Host "  Removed ✓"
}

# Run fresh installer
Write-Host ""
Write-Host "Running fresh install..." -ForegroundColor Cyan
irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.ps1 | iex

# Restore user data
if (Test-Path $BackupDir) {
    Write-Host ""
    Write-Host "Restoring user data..." -ForegroundColor Cyan
    if (Test-Path "$BackupDir\models")      { Copy-Item "$BackupDir\models\*" "$HipfireDir\models\" -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "  models ✓" }
    if (Test-Path "$BackupDir\runtime")     { Copy-Item "$BackupDir\runtime\*" "$HipfireDir\runtime\" -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "  runtime ✓" }
    Remove-Item -Recurse -Force $BackupDir
    Write-Host "  Cleanup ✓"
}

Write-Host ""
Write-Host "=== Force-update complete ===" -ForegroundColor Cyan
