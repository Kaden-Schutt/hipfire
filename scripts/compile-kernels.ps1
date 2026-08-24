# Pre-compile all HIP kernels for target GPU architectures (Windows).
# Usage: .\scripts\compile-kernels.ps1 [arch1 arch2 ...]
# Default: gfx906 gfx1010 gfx1030 gfx1100 gfx1200 gfx1201
#
# Mirror of scripts/compile-kernels.sh. Variant precedence:
#   1. ${name}.${arch}.hip          (chip-specific, e.g. .gfx1100.)
#   2. ${name}.${arch_family}.hip   (family, e.g. .gfx12.)
#   3. ${name}.hip                  (default)

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Archs = @()
)

$ErrorActionPreference = "Stop"

if ($Archs.Count -eq 0) {
    $Archs = @("gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx1201")
}

# Paths: script lives in <repo>\scripts; kernels in <repo>\kernels
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir   = Split-Path -Parent $ScriptDir
$SrcDir    = Join-Path $RepoDir "kernels\src"
$OutBase   = Join-Path $RepoDir "kernels\compiled"

if (-not (Test-Path $SrcDir)) {
    Write-Host "ERROR: kernel source dir not found at $SrcDir" -ForegroundColor Red
    exit 1
}

# Locate hipcc. ROCm 6.4+ on Windows ships hipcc.exe as the preferred entry
# point; older installs ship hipcc.bat (a Perl wrapper). Probe .exe first
# because .bat fails with "filename / directory name / volume label syntax
# is incorrect" on some 6.4 setups (reported by darkamgine in PR #117).
# Fall back to .bat and the no-extension variant for older installs.
function Find-Hipcc {
    $candidates = @()
    if ($env:HIP_PATH) {
        $candidates += @(
            (Join-Path $env:HIP_PATH "bin\hipcc.exe"),
            (Join-Path $env:HIP_PATH "bin\hipcc.bat"),
            (Join-Path $env:HIP_PATH "bin\hipcc")
        )
    }
    $rocmBase = "C:\Program Files\AMD\ROCm"
    if (Test-Path $rocmBase) {
        $verDirs = Get-ChildItem $rocmBase -Directory -ErrorAction SilentlyContinue |
                   Sort-Object Name -Descending
        foreach ($d in $verDirs) {
            $candidates += @(
                (Join-Path $d.FullName "bin\hipcc.exe"),
                (Join-Path $d.FullName "bin\hipcc.bat"),
                (Join-Path $d.FullName "bin\hipcc")
            )
        }
        $candidates += @(
            "C:\Program Files\AMD\ROCm\bin\hipcc.exe",
            "C:\Program Files\AMD\ROCm\bin\hipcc.bat"
        )
    }
    # PATH fallback
    $onPath = Get-Command hipcc -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

$Hipcc = Find-Hipcc
if (-not $Hipcc) {
    Write-Host "ERROR: hipcc not found." -ForegroundColor Red
    Write-Host "  Install the AMD HIP SDK for Windows:"
    Write-Host "    https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
    Write-Host "  Or set `$env:HIP_PATH to your ROCm install (e.g. C:\Program Files\AMD\ROCm\6.4)"
    exit 1
}

Write-Host "=== hipfire kernel compiler ===" -ForegroundColor Cyan
Write-Host "  hipcc: $Hipcc"
Write-Host "  Source: $SrcDir"
Write-Host "  Architectures: $($Archs -join ', ')"

# Variant-tag regex: matches .gfxNNNN. (chip) and .gfxNN. (family).
$VariantTagRe = '\.gfx[0-9]+\.hip$'

$Total  = 0
$Failed = 0

foreach ($arch in $Archs) {
    $outDir = Join-Path $OutBase $arch
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host ""
    Write-Host "--- $arch ---"

    # Family tag: first 5 chars of arch ("gfx12", "gfx10", etc.).
    $archFamily = $arch.Substring(0, [Math]::Min(5, $arch.Length))

    foreach ($src in (Get-ChildItem -Path $SrcDir -Filter "*.hip" -File)) {
        $base = $src.Name

        # Skip variant-tagged files during the parent iteration.
        if ($base -match $VariantTagRe) { continue }

        $name = [System.IO.Path]::GetFileNameWithoutExtension($base)

        # gfx906 (Vega 20 / GCN5) lacks WMMA + dot8; skip those kernels.
        if ($arch -eq "gfx906") {
            if ($name -match '_wmma' -or $name -eq "gemv_mq8g256") {
                Write-Host "  - $name SKIP (unsupported ISA on gfx906)"
                continue
            }
        }

        # RDNA3 (gfx11) lacks dot1-insts: every sdot4/dp4a kernel cannot build.
        if ($archFamily -eq "gfx11" -and ($name -eq "gemv_mq8g256" -or $name -match '_dp4a$')) {
            Write-Host "  - $name SKIP (dot1-insts unavailable on $arch)"
            continue
        }

        # fwht3 hd512 KV/attention kernels reference fwht/turbo *_512 helpers
        # that do not exist in turbo_common.h yet (upstream TODO). Other
        # _hd512 kernels (e.g. asym_k_givens3) build fine and stay.
        if ($name -match '^(attention_flash_fwht3_tile|kv_cache_write_fwht3)_hd512') {
            Write-Host "  - $name SKIP (missing _512 helpers in turbo_common.h)"
            continue
        }

        # Temporal-buffer GEMVs are opt-in routes assembled by the JIT with
        # CACHE_ELIGIBLE/OPT_IN defines; without them the source cannot link
        # against ordinary pointer loads. Not part of the default dispatch.
        if ($name -eq "gemv_mfp4g32_e8_soa_u4_buffer") {
            Write-Host "  - $name SKIP (opt-in buffer route; JIT-only)"
            continue
        }

        # Skip sources whose name pins them to a different chip
        # (e.g. *_gfx1151, *_gfx906_*, *_gfx12_*) unless the embedded arch is
        # this arch or its family (gfx11_dgpu style variants stay).
        if ($name -match '[._]gfx(\d+)') {
            $pinned = $Matches[1]
            $archNum = $arch -replace '^gfx', ''
            $famNum = $archFamily -replace '^gfx', ''
            if ($pinned -ne $archNum -and $pinned -ne $famNum) {
                Write-Host "  - $name SKIP (pinned to gfx$pinned)"
                continue
            }
        }

        # Variant precedence
        $chipVariant   = Join-Path $SrcDir "$name.$arch.hip"
        $familyVariant = Join-Path $SrcDir "$name.$archFamily.hip"
        $srcPath = $src.FullName
        if (Test-Path $chipVariant) {
            $srcPath = $chipVariant
            Write-Host "  [variant] $name ($arch chip-specific)"
        } elseif (Test-Path $familyVariant) {
            $srcPath = $familyVariant
            Write-Host "  [variant] $name ($archFamily family)"
        }

        $outPath = Join-Path $outDir "$name.hsaco"
        $Total++

        # Emulate the JIT's source assembly (crates/rdna-compute/src/kernels.rs
        # concat!): HFQ4 sources reference the HIPFIRE_WEIGHT_* macros that
        # gfx12_weight_cache_policy.inc provides, and kernels.rs prepends that
        # preamble ahead of them. The standalone script must do the same or the
        # compile fails with undeclared identifiers even though the source is
        # valid. Only prepend when the file uses the macros but does not define
        # its own (avoids macro-redefinition in self-contained files).
        $compilePath = $srcPath
        $srcText = [System.IO.File]::ReadAllText($srcPath)
        if (($srcText -match 'HIPFIRE_WEIGHT_(LOAD|RSRC)' -or
             $srcText -match 'HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS') -and
            ($srcText -notmatch '#define HIPFIRE_WEIGHT_') -and
            ($srcText -notmatch 'weight_cache_policy\.inc')) {
            $incPath = Join-Path $SrcDir "gfx12_weight_cache_policy.inc"
            if (Test-Path $incPath) {
                $combined = [System.IO.File]::ReadAllText($incPath) + "`n" + $srcText
                $compilePath = Join-Path $SrcDir "_concat_$base"
                [System.IO.File]::WriteAllText($compilePath, $combined)
            }
        }

        $args = @(
            "--genco",
            "--offload-arch=$arch",
            "-O3",
            "-I", "`"$SrcDir`"",
            "-o", "`"$outPath`"",
            "`"$compilePath`""
        )

        # On Windows hipcc is typically a .bat; invoke via cmd /c so PowerShell
        # routes args correctly.
        if ($Hipcc.ToLower().EndsWith(".bat")) {
            $cmdLine = "`"$Hipcc`" " + ($args -join " ")
            $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmdLine `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardError "$outPath.err"
        } else {
            $proc = Start-Process -FilePath $Hipcc -ArgumentList $args `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardError "$outPath.err"
        }

        if ($proc.ExitCode -eq 0 -and (Test-Path $outPath)) {
            $sizeKB = [int]((Get-Item $outPath).Length / 1024)
            Write-Host "  OK $name ($sizeKB KB)" -ForegroundColor Green
            Remove-Item -Path "$outPath.err" -Force -ErrorAction SilentlyContinue
        } else {
            Write-Host "  FAIL $name" -ForegroundColor Red
            $Failed++
            Remove-Item -Path $outPath -Force -ErrorAction SilentlyContinue
            if (Test-Path "$outPath.err") {
                $errText = Get-Content "$outPath.err" -Raw -ErrorAction SilentlyContinue
                if ($errText) { Write-Host "    $($errText.Trim())" -ForegroundColor DarkGray }
                Remove-Item -Path "$outPath.err" -Force -ErrorAction SilentlyContinue
            }
        }

        if ($compilePath -ne $srcPath) {
            Remove-Item -Path $compilePath -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host ""
$ok = $Total - $Failed
Write-Host "=== Done: $ok/$Total compiled, $Failed failed ===" -ForegroundColor Cyan
if ($Failed -gt 0) { exit 1 }
