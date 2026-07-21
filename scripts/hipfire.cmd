@echo off
setlocal
if exist "%~dp0..\target\release\hipfire.exe" (
  "%~dp0..\target\release\hipfire.exe" %*
  exit /b %ERRORLEVEL%
)
if exist "%USERPROFILE%\.hipfire\bin\hipfire.exe" (
  "%USERPROFILE%\.hipfire\bin\hipfire.exe" %*
  exit /b %ERRORLEVEL%
)
cargo run --quiet --manifest-path "%~dp0..\Cargo.toml" -p hipfire-cli -- %*
