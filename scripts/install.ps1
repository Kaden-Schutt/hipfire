# hipfire installer for Windows — detects GPU, installs deps, downloads binary + kernels.
# Usage: irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.ps1 | iex
$ErrorActionPreference = "Stop"

# ─── Paths ───────────────────────────────────────────────
$HipfireDir  = "$env:USERPROFILE\.hipfire"
$BinDir      = "$HipfireDir\bin"
$RuntimeDir  = "$HipfireDir\runtime"
$ModelsDir   = "$HipfireDir\models"
$SrcDir      = "$HipfireDir\src"

# ─── Constants ───────────────────────────────────────────
$GithubRepo   = "Kaden-Schutt/hipfire"
$GithubBranch = "master"

Write-Host "=== hipfire installer ===" -ForegroundColor Cyan
Write-Host ""

# ─── GPU Detection ───────────────────────────────────────
Write-Host "Checking for AMD GPU..." -ForegroundColor Cyan

$GpuArch = "unknown"
try {
    $VideoControllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
    $AmdGpu = $VideoControllers | Where-Object { $_.Name -match "AMD|Radeon" } | Select-Object -First 1
    if ($AmdGpu) {
        $GpuName = $AmdGpu.Name
        Write-Host "  Found: $GpuName"

        # Map GPU name to arch
        if ($GpuName -match "5700|RX 5[0-9]{3}") {
            $GpuArch = "gfx1010"
        } elseif ($GpuName -match "6[89]00|6[79]50|6[89]50|RX 6[0-9]{3}") {
            $GpuArch = "gfx1030"
        } elseif ($GpuName -match "7900|7800|7700|7600|RX 7[0-9]{3}") {
            $GpuArch = "gfx1100"
        } elseif ($GpuName -match "9070") {
            $GpuArch = "gfx1201"
        } elseif ($GpuName -match "9060|RX 9[0-9]{3}") {
            $GpuArch = "gfx1200"
        }
    } else {
        Write-Host "  WARNING: No AMD/Radeon GPU found in Win32_VideoController." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  WARNING: Could not query GPU information: $_" -ForegroundColor Yellow
}

if ($GpuArch -eq "unknown") {
    Write-Host "  WARNING: Could not detect GPU architecture." -ForegroundColor Yellow
    Write-Host "  Supported: gfx1010 (RX 5700), gfx1030 (RX 6800), gfx1100 (RX 7900), gfx1200 (RX 9060), gfx1201 (RX 9070)"
    $GpuArch = Read-Host "  Enter your GPU arch [or Enter to skip]"
    if ([string]::IsNullOrWhiteSpace($GpuArch)) { $GpuArch = "unknown" }
}
Write-Host "  GPU arch: $GpuArch" -ForegroundColor Green

# ─── Create directories ──────────────────────────────────
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $BinDir    | Out-Null
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir  | Out-Null
Write-Host "  $BinDir" -ForegroundColor Green
Write-Host "  $RuntimeDir" -ForegroundColor Green
Write-Host "  $ModelsDir" -ForegroundColor Green

# ─── HIP DLL (amdhip64.dll) ──────────────────────────────
Write-Host ""
Write-Host "Checking HIP runtime (amdhip64.dll)..." -ForegroundColor Cyan

$HipDllFound = $false
$HipDllDest  = "$RuntimeDir\amdhip64.dll"

# Check RuntimeDir first (idempotent re-runs)
if (Test-Path $HipDllDest) {
    Write-Host "  amdhip64.dll: found in RuntimeDir ✓" -ForegroundColor Green
    $HipDllFound = $true
}

# Check %HIP_PATH%\bin (unversioned and versioned)
if (-not $HipDllFound -and $env:HIP_PATH) {
    foreach ($dllName in @("amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll")) {
        $candidate = Join-Path $env:HIP_PATH "bin\$dllName"
        if (Test-Path $candidate) {
            Write-Host "  ${dllName}: found at $candidate ✓" -ForegroundColor Green
            Copy-Item $candidate $HipDllDest -Force
            $HipDllFound = $true
            break
        }
    }
}

# Check standard ROCm install locations (unversioned and versioned)
if (-not $HipDllFound) {
    foreach ($dllName in @("amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll")) {
        # Check versioned ROCm dirs (e.g. C:\Program Files\AMD\ROCm\7.1\bin\)
        $rocmBase = "C:\Program Files\AMD\ROCm"
        if (Test-Path $rocmBase) {
            foreach ($verDir in (Get-ChildItem $rocmBase -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending)) {
                $candidate = Join-Path $verDir.FullName "bin\$dllName"
                if (Test-Path $candidate) {
                    Write-Host "  ${dllName}: found at $candidate ✓" -ForegroundColor Green
                    Copy-Item $candidate $HipDllDest -Force
                    $HipDllFound = $true
                    break
                }
            }
            if ($HipDllFound) { break }
        }
        # Also check flat layout
        $candidate = "C:\Program Files\AMD\ROCm\bin\$dllName"
        if (Test-Path $candidate) {
            Write-Host "  ${dllName}: found at $candidate ✓" -ForegroundColor Green
            Copy-Item $candidate $HipDllDest -Force
            $HipDllFound = $true
            break
        }
    }
}

# Attempt download from GitHub release
if (-not $HipDllFound) {
    Write-Host "  amdhip64.dll: not found locally. Downloading from GitHub release..." -ForegroundColor Yellow
    $DllUrl = "https://github.com/$GithubRepo/releases/download/hip-runtime/amdhip64.dll"
    try {
        Invoke-WebRequest -Uri $DllUrl -OutFile $HipDllDest -UseBasicParsing
        Write-Host "  amdhip64.dll: downloaded ✓" -ForegroundColor Green
        $HipDllFound = $true
    } catch {
        Write-Host "  amdhip64.dll: download failed: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "  hipfire needs amdhip64.dll to run. Install ROCm for Windows manually:" -ForegroundColor Yellow
        Write-Host "    https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html"
        Write-Host "  Or place amdhip64.dll in: $RuntimeDir"
        Write-Host ""
        $reply = Read-Host "  Continue without HIP runtime? [y/N]"
        if ($reply -notmatch "^[Yy]$") {
            Write-Host "Exiting. Re-run after installing ROCm." -ForegroundColor Red
            exit 1
        }
    }
}

# Ensure runtime dir is in PATH for this session so daemon can find the DLL
if ($HipDllFound) {
    $env:PATH = "$RuntimeDir;$env:PATH"
}

# ─── HIP version vs GPU arch check ──────────────────────
if ($HipDllFound -and $GpuArch -ne "unknown") {
    # Try to get HIP version from the DLL or hipconfig
    $HipVer = ""
    $hipconfig = "$env:HIP_PATH\bin\hipconfig.exe"
    if (-not (Test-Path $hipconfig)) { $hipconfig = "C:\Program Files\AMD\ROCm\bin\hipconfig.exe" }
    if (Test-Path $hipconfig) {
        try { $HipVer = (& $hipconfig --version 2>$null) -replace '[^\d.]','' | Select-Object -First 1 } catch {}
    }
    # Fallback: check DLL file version
    if (-not $HipVer) {
        try {
            $dllPath = if (Test-Path $HipDllDest) { $HipDllDest } else { $candidate }
            $ver = (Get-Item $dllPath).VersionInfo.ProductVersion
            if ($ver) { $HipVer = $ver }
        } catch {}
    }

    if ($HipVer) {
        $parts = $HipVer.Split(".")
        $major = [int]$parts[0]
        $minor = if ($parts.Length -gt 1) { [int]$parts[1] } else { 0 }
        Write-Host "  HIP version: $major.$minor" -ForegroundColor Green

        # Minimum versions per arch
        $minMajor = 5; $minMinor = 0
        switch ($GpuArch) {
            { $_ -in "gfx1200","gfx1201" } { $minMajor = 6; $minMinor = 4 }
            { $_ -in "gfx1100","gfx1101" } { $minMajor = 5; $minMinor = 5 }
        }

        if ($major -lt $minMajor -or ($major -eq $minMajor -and $minor -lt $minMinor)) {
            Write-Host ""
            Write-Host "  WARNING: HIP $major.$minor is too old for $GpuArch (needs $minMajor.$minMinor+)" -ForegroundColor Red
            Write-Host "  Kernels may fail to load. Update AMD HIP SDK:" -ForegroundColor Yellow
            Write-Host "    https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html" -ForegroundColor Yellow
            Write-Host ""
            $reply = Read-Host "  Continue anyway? [y/N]"
            if ($reply -notmatch "^[Yy]$") { exit 1 }
        }
    }
}

# ─── Bun (CLI runtime) ───────────────────────────────────
Write-Host ""
Write-Host "Checking Bun..." -ForegroundColor Cyan

$BunBin = "$env:USERPROFILE\.bun\bin"
if (Get-Command bun -ErrorAction SilentlyContinue) {
    Write-Host "  Bun: found ✓" -ForegroundColor Green
} else {
    Write-Host "  Bun not found. Installing..." -ForegroundColor Yellow
    try {
        powershell -c "irm bun.sh/install.ps1 | iex"
        # Add bun to PATH for remainder of this session
        $env:PATH = "$BunBin;$env:PATH"
        if (Get-Command bun -ErrorAction SilentlyContinue) {
            Write-Host "  Bun installed ✓" -ForegroundColor Green
        } else {
            Write-Host "  Bun installed but not in PATH. Add manually:" -ForegroundColor Yellow
            Write-Host "    $BunBin"
        }
    } catch {
        Write-Host "  Bun install failed: $_" -ForegroundColor Red
        Write-Host "  Visit https://bun.sh and install manually, then re-run."
        exit 1
    }
}

# ─── Clone / update repo ─────────────────────────────────
Write-Host ""
Write-Host "Setting up hipfire source..." -ForegroundColor Cyan

if (-not (Test-Path "$SrcDir\.git")) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "  ERROR: git is required. Install from https://git-scm.com and re-run." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Cloning https://github.com/$GithubRepo.git ..."
    try {
        git clone --depth 1 --branch $GithubBranch "https://github.com/$GithubRepo.git" $SrcDir
        Write-Host "  Cloned ✓" -ForegroundColor Green
    } catch {
        Write-Host "  Clone failed: $_" -ForegroundColor Red
        Write-Host "  Try manually: git clone https://github.com/$GithubRepo.git $SrcDir"
        exit 1
    }
} else {
    Write-Host "  Existing clone found at $SrcDir"
    $status = & git -C $SrcDir status --porcelain 2>&1 | Out-String
    if ($status.Trim()) {
        Write-Host "  WARNING: local modifications detected." -ForegroundColor Yellow
        $reply = Read-Host "  Overwrite local changes and update? [y/N]"
        if ($reply -match "^[Yy]$") {
            try {
                & git -C $SrcDir fetch origin $GithubBranch --depth 1 2>&1 | Out-Null
                & git -C $SrcDir reset --hard "origin/$GithubBranch" 2>&1 | Out-Null
                Write-Host "  Updated ✓" -ForegroundColor Green
            } catch {
                Write-Host "  Update failed (non-fatal)." -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Keeping existing checkout." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Updating..."
        try {
            $env:GIT_TERMINAL_PROMPT = "0"
            & git -C $SrcDir fetch origin $GithubBranch --depth 1 2>&1 | Out-Null
            & git -C $SrcDir reset --hard "origin/$GithubBranch" 2>&1 | Out-Null
            Write-Host "  Updated ✓" -ForegroundColor Green
        } catch {
            Write-Host "  Update failed (non-fatal). Using existing checkout." -ForegroundColor Yellow
        }
    }
}

$RepoDir = $SrcDir

# ─── Build / install binaries ────────────────────────────
Write-Host ""
Write-Host "Installing hipfire binaries..." -ForegroundColor Cyan

# Look for pre-built daemon.exe: local first, then download from GitHub release
$PreBuilt = @(
    "$RepoDir\target\release\examples\daemon.exe",
    "$RepoDir\bin\daemon.exe",
    "$BinDir\daemon.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $PreBuilt) {
    # Download pre-built Windows binary from GitHub release
    Write-Host "  Downloading pre-built daemon.exe from GitHub release..."
    $ReleaseUrl = "https://github.com/$GithubRepo/releases/download/v0.1.0-alpha/daemon.exe"
    try {
        Invoke-WebRequest -Uri $ReleaseUrl -OutFile "$BinDir\daemon.exe" -UseBasicParsing
        $PreBuilt = "$BinDir\daemon.exe"
        Write-Host "  Downloaded ✓" -ForegroundColor Green
    } catch {
        Write-Host "  Download failed: $_" -ForegroundColor Yellow
    }
}

if ($PreBuilt -and $PreBuilt -ne "$BinDir\daemon.exe") {
    Copy-Item $PreBuilt "$BinDir\daemon.exe" -Force
    Write-Host "  daemon.exe installed ✓" -ForegroundColor Green
} elseif ($PreBuilt) {
    Write-Host "  daemon.exe ready ✓" -ForegroundColor Green
} else {
    Write-Host "  No pre-built binaries available. Building from source..." -ForegroundColor Yellow

    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Host "  Installing Rust via rustup..." -ForegroundColor Yellow
        $RustupUrl  = "https://win.rustup.rs/x86_64"
        $RustupExe  = "$env:TEMP\rustup-init.exe"
        Invoke-WebRequest -Uri $RustupUrl -OutFile $RustupExe -UseBasicParsing
        & $RustupExe -y --default-toolchain stable
        # Add cargo to PATH for this session
        $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    }

    Write-Host "  cargo build --release (this may take several minutes)..."
    Push-Location $RepoDir
    try {
        cargo build --release --features deltanet --example daemon --example infer --example infer_hfq -p engine
    } finally {
        Pop-Location
    }

    $BuiltExe = "$RepoDir\target\release\examples\daemon.exe"
    if (-not (Test-Path $BuiltExe)) {
        Write-Host ""
        Write-Host "  BUILD FAILED." -ForegroundColor Red
        Write-Host "  Common causes:"
        Write-Host "    - Missing ROCm SDK (needed to compile)"
        Write-Host "    - Missing Visual C++ build tools"
        Write-Host ""
        Write-Host "  After fixing, re-run this installer or build manually:"
        Write-Host "    cd $RepoDir"
        Write-Host "    cargo build --release --features deltanet --example daemon -p engine"
        exit 1
    }
    Copy-Item $BuiltExe "$BinDir\daemon.exe" -Force
    Write-Host "  Build complete ✓" -ForegroundColor Green
}

# Copy optional helper binaries if present
foreach ($exe in @("infer.exe", "infer_hfq.exe")) {
    $src = "$RepoDir\target\release\examples\$exe"
    if (Test-Path $src) { Copy-Item $src "$BinDir\$exe" -Force }
}

# ─── CLI ─────────────────────────────────────────────────
Write-Host ""
Write-Host "Installing CLI..." -ForegroundColor Cyan

$CliDir = "$HipfireDir\cli"
New-Item -ItemType Directory -Force -Path $CliDir | Out-Null
Copy-Item "$RepoDir\cli\index.ts"    "$CliDir\index.ts"    -Force
Copy-Item "$RepoDir\cli\package.json" "$CliDir\package.json" -Force

# Create hipfire.cmd wrapper
$CmdWrapper = "@echo off`r`nbun run `"%USERPROFILE%\.hipfire\cli\index.ts`" %*`r`n"
[System.IO.File]::WriteAllText("$BinDir\hipfire.cmd", $CmdWrapper)

Write-Host "  CLI installed to $CliDir ✓" -ForegroundColor Green
Write-Host "  Wrapper: $BinDir\hipfire.cmd ✓" -ForegroundColor Green

# ─── Kernels ─────────────────────────────────────────────
Write-Host ""
if ($GpuArch -ne "unknown") {
    Write-Host "Setting up kernels for $GpuArch..." -ForegroundColor Cyan
    $KernelSrc  = "$RepoDir\kernels\compiled\$GpuArch"
    $KernelDest = "$BinDir\kernels\compiled\$GpuArch"
    New-Item -ItemType Directory -Force -Path $KernelDest | Out-Null

    if (Test-Path $KernelSrc) {
        $Hsacos = Get-ChildItem "$KernelSrc\*.hsaco" -ErrorAction SilentlyContinue
        if ($Hsacos.Count -gt 0) {
            Copy-Item "$KernelSrc\*.hsaco" $KernelDest -Force
            Copy-Item "$KernelSrc\*.hash" $KernelDest -Force -ErrorAction SilentlyContinue
            Write-Host "  Copied $($Hsacos.Count) kernels + hashes to $KernelDest ✓" -ForegroundColor Green
        } else {
            Write-Host "  WARNING: No .hsaco files found in $KernelSrc" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  WARNING: No pre-compiled kernels for $GpuArch in repo." -ForegroundColor Yellow
        Write-Host "  Compile them: cd $RepoDir && scripts\compile-kernels.ps1 $GpuArch"
        Write-Host "  Then copy: Copy-Item kernels\compiled\$GpuArch\*.hsaco $KernelDest"
    }
} else {
    Write-Host "Skipping kernel setup (GPU arch unknown)." -ForegroundColor Yellow
    Write-Host "  Re-run installer after fixing GPU detection, or copy kernels manually."
}

# ─── Config ──────────────────────────────────────────────
$ConfigFile = "$HipfireDir\config.json"
if (-not (Test-Path $ConfigFile)) {
    $Config = [ordered]@{
        temperature = 0.3
        top_p       = 0.8
        max_tokens  = 512
        gpu_arch    = $GpuArch
    } | ConvertTo-Json
    [System.IO.File]::WriteAllText($ConfigFile, $Config)
    Write-Host ""
    Write-Host "Config written: $ConfigFile" -ForegroundColor Green
}

# ─── PATH ────────────────────────────────────────────────
Write-Host ""
$NoPath = $args -contains "--no-path"
$CurrentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($null -eq $CurrentUserPath) { $CurrentUserPath = "" }

if ($NoPath) {
    Write-Host "Skipping PATH modification (--no-path)" -ForegroundColor Yellow
    Write-Host "  Add manually to user PATH: $BinDir" -ForegroundColor Yellow
} elseif ($CurrentUserPath -notlike "*$BinDir*") {
    Write-Host "hipfire bin dir is not in your user PATH." -ForegroundColor Yellow
    Write-Host "  $BinDir"
    $reply = Read-Host "Add to user PATH permanently? [Y/n]"
    if ($reply -notmatch "^[Nn]$") {
        $NewPath = "$BinDir;$CurrentUserPath"
        # Safety: warn if PATH would exceed Windows limit (2047 chars)
        if ($NewPath.Length -gt 2040) {
            Write-Host "  WARNING: User PATH would be $($NewPath.Length) chars (limit ~2047)." -ForegroundColor Red
            Write-Host "  Skipping to avoid PATH truncation. Add manually:" -ForegroundColor Red
            Write-Host "    $BinDir" -ForegroundColor Yellow
        } else {
            [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
            $env:PATH = "$BinDir;$env:PATH"
            Write-Host "  PATH updated ✓ (restart your shell to apply)" -ForegroundColor Green
        }
    } else {
        Write-Host "  Add manually to user PATH: $BinDir" -ForegroundColor Yellow
    }
} else {
    Write-Host "hipfire already in PATH ✓" -ForegroundColor Green
}

# ─── Quick start ─────────────────────────────────────────
Write-Host ""
Write-Host "=== hipfire installed ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Green
Write-Host "  hipfire list                        # see local models"
Write-Host "  hipfire run <model.hfq> `"Hello`"    # generate text"
Write-Host "  hipfire serve                       # start OpenAI-compatible API"
Write-Host ""
Write-Host "Models go in $ModelsDir"
Write-Host ""
