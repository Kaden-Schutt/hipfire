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

# Resolve cargo's actual target directory. Honors CARGO_TARGET_DIR and any
# workspace target overrides — without this, users with a shared target
# directory (common when juggling several Rust projects) would see install
# fail because the binaries we expect at $RepoDir\target\... actually live
# elsewhere. Falls back to the conventional location when cargo is not yet
# installed, since pre-built binaries can only sit at the default path in
# that case.
$TargetDir = "$RepoDir\target"
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    try {
        $Meta = cargo metadata --format-version 1 --manifest-path "$RepoDir\Cargo.toml" 2>$null | ConvertFrom-Json
        if ($Meta.target_directory) { $TargetDir = $Meta.target_directory }
    } catch {}
}

# Source preference order — the prior install's $BinDir\daemon.exe is NOT
# a source. Including it would re-use stale binaries forever. The repo-side
# paths (only meaningful when running install.ps1 from a checkout) are
# treated as developer-authoritative and used if present; everyone else
# always pulls the latest release asset.
$PreBuilt = @(
    "$TargetDir\release\examples\daemon.exe",
    "$RepoDir\bin\daemon.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $PreBuilt) {
    # Query the latest GitHub release dynamically — never pin to an old tag.
    # If the latest release has a daemon.exe asset, use it; otherwise fall
    # through to the source-build path below. This way every Windows install
    # tracks current master without requiring a script bump per release.
    Write-Host "  Querying latest GitHub release..."
    try {
        $LatestRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/$GithubRepo/releases/latest" -UseBasicParsing
        $DaemonAsset = $LatestRelease.assets | Where-Object { $_.name -eq "daemon.exe" } | Select-Object -First 1
        if ($DaemonAsset) {
            $ReleaseDest = "$BinDir\daemon.exe"
            # Cache discipline: store the asset id (immutable per upload) of
            # whatever we last downloaded as a sidecar. Refresh whenever the
            # current asset id differs — catches both new tags and re-uploaded
            # binaries on the same tag. Size-only matching was insufficient
            # because two builds can share a byte count.
            $StampPath = "$BinDir\daemon.exe.release-id"
            $CurrentAssetId = "$($DaemonAsset.id)"
            $NeedsDownload = $true
            if ((Test-Path $ReleaseDest) -and (Test-Path $StampPath)) {
                $StoredAssetId = (Get-Content $StampPath -Raw -ErrorAction SilentlyContinue).Trim()
                if ($StoredAssetId -eq $CurrentAssetId) {
                    Write-Host "  daemon.exe already at latest release $($LatestRelease.tag_name) (asset id $CurrentAssetId) — keeping" -ForegroundColor Green
                    $PreBuilt = $ReleaseDest
                    $NeedsDownload = $false
                } else {
                    Write-Host "  daemon.exe is stale (local id $StoredAssetId, latest id $CurrentAssetId) — refreshing" -ForegroundColor Yellow
                }
            } elseif (Test-Path $ReleaseDest) {
                Write-Host "  daemon.exe present but no release-id stamp — refreshing to be safe" -ForegroundColor Yellow
            }
            if ($NeedsDownload) {
                Write-Host "  Pulling daemon.exe from release $($LatestRelease.tag_name)..." -ForegroundColor Cyan
                $TempDest = "$ReleaseDest.download"
                try {
                    # Stage to a temp file so a partial / failed download
                    # doesn't corrupt the existing daemon.exe.
                    Invoke-WebRequest -Uri $DaemonAsset.browser_download_url -OutFile $TempDest -UseBasicParsing
                    if ((Get-Item $TempDest).Length -ne $DaemonAsset.size) {
                        Remove-Item $TempDest -Force -ErrorAction SilentlyContinue
                        throw "downloaded size mismatch (got $((Get-Item $TempDest).Length), expected $($DaemonAsset.size))"
                    }
                    Move-Item -Path $TempDest -Destination $ReleaseDest -Force
                    Set-Content -Path $StampPath -Value $CurrentAssetId -NoNewline
                    $PreBuilt = $ReleaseDest
                    Write-Host "  Downloaded ✓" -ForegroundColor Green
                } catch {
                    Remove-Item $TempDest -Force -ErrorAction SilentlyContinue
                    Write-Host "  Download failed: $_" -ForegroundColor Yellow
                    Write-Host "  Existing daemon.exe (if any) is unchanged. Falling through to source build." -ForegroundColor Yellow
                }
            }
        } else {
            Write-Host "  No daemon.exe in release $($LatestRelease.tag_name) — falling through to source build" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  Could not query GitHub release API: $_ — falling through to source build" -ForegroundColor Yellow
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
        cargo build --release --features deltanet --example daemon --example infer --example infer_hfq -p hipfire-runtime
    } finally {
        Pop-Location
    }

    # Re-resolve TargetDir now that cargo is guaranteed available — covers the
    # case where rustup was just installed above and the initial probe fell
    # back to the default path.
    try {
        $Meta = cargo metadata --format-version 1 --manifest-path "$RepoDir\Cargo.toml" 2>$null | ConvertFrom-Json
        if ($Meta.target_directory) { $TargetDir = $Meta.target_directory }
    } catch {}

    $BuiltExe = "$TargetDir\release\examples\daemon.exe"
    if (-not (Test-Path $BuiltExe)) {
        Write-Host ""
        Write-Host "  BUILD FAILED." -ForegroundColor Red
        Write-Host "  Common causes:"
        Write-Host "    - Missing ROCm SDK (needed to compile)"
        Write-Host "    - Missing Visual C++ build tools"
        Write-Host ""
        Write-Host "  After fixing, re-run this installer or build manually:"
        Write-Host "    cd $RepoDir"
        Write-Host "    cargo build --release --features deltanet --example daemon -p hipfire-runtime"
        exit 1
    }
    Copy-Item $BuiltExe "$BinDir\daemon.exe" -Force
    Write-Host "  Build complete ✓" -ForegroundColor Green
}

# Copy optional helper binaries if present
foreach ($exe in @("infer.exe", "infer_hfq.exe")) {
    $src = "$TargetDir\release\examples\$exe"
    if (Test-Path $src) { Copy-Item $src "$BinDir\$exe" -Force }
}

# ─── CLI ─────────────────────────────────────────────────
Write-Host ""
Write-Host "Installing CLI..." -ForegroundColor Cyan

$CliDir = "$HipfireDir\cli"
New-Item -ItemType Directory -Force -Path $CliDir | Out-Null
# Recursive copy of the whole cli\ directory, then prune dev/test artifacts.
# New .ts files added to cli\ (next chat helper, future slash-command module)
# are picked up automatically — no install-script edit required. Replaces
# the previous per-file enumeration that grew stale after PR #129 added
# chat.ts/chat_pure.ts (issue #163, patched in #165; this is the structural
# follow-up that PR left for later).
if (-not (Test-Path "$RepoDir\cli\registry.json") -or -not (Test-Path "$RepoDir\cli\index.ts")) {
    Write-Host "ERROR: cli\registry.json or cli\index.ts missing in $RepoDir" -ForegroundColor Red
    Write-Host "       Repo checkout may be incomplete; aborting install." -ForegroundColor Red
    exit 1
}
# Robocopy mirrors better than Copy-Item -Recurse for this case (handles
# permissions, exit-code semantics, and is on every Windows installation),
# but Copy-Item is more portable across PowerShell core / Windows PS / pwsh
# on macOS-via-PS-remoting; sticking with Copy-Item for parity with the rest
# of the script.
Copy-Item "$RepoDir\cli\*" $CliDir -Recurse -Force
# Prune dev artifacts. Patterns mirror install.sh — tests follow
# `*.test.ts` / `test_*.ts` / `bench_*.ts` Bun conventions; node_modules
# and dotfiles are dev-only. Adding a new test file with the same naming
# requires no install-script change.
$prunePaths = @("node_modules", ".gitignore", "tsconfig.json", "README.md", "bun.lock")
foreach ($p in $prunePaths) {
    $target = Join-Path $CliDir $p
    if (Test-Path $target) { Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue }
}
Get-ChildItem -Path $CliDir -File | Where-Object {
    $_.Name -like "*.test.ts" -or $_.Name -like "test_*.ts" -or $_.Name -like "bench_*.ts"
} | Remove-Item -Force -ErrorAction SilentlyContinue

# Create hipfire.cmd wrapper
$CmdWrapper = "@echo off`r`nbun run `"%USERPROFILE%\.hipfire\cli\index.ts`" %*`r`n"
[System.IO.File]::WriteAllText("$BinDir\hipfire.cmd", $CmdWrapper)

Write-Host "  CLI installed to $CliDir ✓" -ForegroundColor Green
Write-Host "  Wrapper: $BinDir\hipfire.cmd ✓" -ForegroundColor Green

# ─── Kernels ─────────────────────────────────────────────
# kernels/compiled/<arch>/ is gitignored, so a fresh git clone never ships
# .hsaco blobs. We mirror the Linux flow (install.sh): seed any blobs that
# happen to be present in the checkout (developer case), then run
# daemon.exe --precompile to JIT-compile the default Qwen3.5 kernel set
# into ~/.hipfire/bin/kernels/compiled/<arch>/. First `hipfire run` is then
# instant instead of a multi-minute hipcc wall.
Write-Host ""
if ($GpuArch -ne "unknown") {
    Write-Host "Setting up kernels for $GpuArch..." -ForegroundColor Cyan
    $KernelSrc  = "$RepoDir\kernels\compiled\$GpuArch"
    $KernelDest = "$BinDir\kernels\compiled\$GpuArch"
    New-Item -ItemType Directory -Force -Path $KernelDest | Out-Null

    if (Test-Path $KernelSrc) {
        $Hsacos = Get-ChildItem "$KernelSrc\*.hsaco" -ErrorAction SilentlyContinue
        if ($Hsacos -and $Hsacos.Count -gt 0) {
            Copy-Item "$KernelSrc\*.hsaco" $KernelDest -Force
            Copy-Item "$KernelSrc\*.hash" $KernelDest -Force -ErrorAction SilentlyContinue
            Write-Host "  Seeded $($Hsacos.Count) kernels from repo checkout to $KernelDest ✓" -ForegroundColor Green
        } else {
            Write-Host "  No pre-compiled .hsaco found in repo (gitignored). Will JIT-compile below." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  No pre-compiled kernels for $GpuArch in repo (gitignored). Will JIT-compile below." -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping kernel setup (GPU arch unknown)." -ForegroundColor Yellow
    Write-Host "  Re-run installer after fixing GPU detection, or run scripts\compile-kernels.ps1 manually."
}

# ─── Pre-compile via daemon (parity with install.sh) ─────
# Fills in any missing kernels for the active GPU. Uses hipcc in the
# background; writes back to ~/.hipfire/bin/kernels/compiled/<arch>/.
# Runs even when GpuArch is "unknown"; Gpu::init resolves the active arch
# at runtime regardless of install-time detection.
$DaemonExe = "$BinDir\daemon.exe"
if (Test-Path $DaemonExe) {
    Write-Host ""
    Write-Host "Pre-compiling GPU kernels (first run will be instant afterward)..." -ForegroundColor Cyan
    $hipccAvailable = $false
    if ($env:HIP_PATH -and (Test-Path (Join-Path $env:HIP_PATH "bin\hipcc.bat"))) { $hipccAvailable = $true }
    elseif ($env:HIP_PATH -and (Test-Path (Join-Path $env:HIP_PATH "bin\hipcc.exe"))) { $hipccAvailable = $true }
    elseif (Get-Command hipcc -ErrorAction SilentlyContinue) { $hipccAvailable = $true }
    elseif (Test-Path "C:\Program Files\AMD\ROCm") {
        $rocmHipcc = Get-ChildItem "C:\Program Files\AMD\ROCm" -Recurse -Filter "hipcc.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($rocmHipcc) { $hipccAvailable = $true }
    }

    if (-not $hipccAvailable) {
        Write-Host "  hipcc not found in PATH or `$env:HIP_PATH; skipping pre-compile." -ForegroundColor Yellow
        Write-Host "  Install the AMD HIP SDK to enable JIT compilation:" -ForegroundColor Yellow
        Write-Host "    https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html" -ForegroundColor Yellow
        Write-Host "  Pre-compiled blobs in the repo will still load if available."
    } else {
        try {
            & $DaemonExe --precompile
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  Pre-compile complete ✓" -ForegroundColor Green
            } else {
                Write-Host "  Pre-compile finished with warnings; missing kernels will JIT on first use." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  Pre-compile failed: $_; missing kernels will JIT on first use." -ForegroundColor Yellow
        }
    }
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
