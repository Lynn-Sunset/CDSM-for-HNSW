# Phase 1: upper-layer restart oracle analysis on chunks-cx-paper-v1
# Runs both vector fields.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$jdk = 'C:\elasticsearch-8.17.0\jdk\bin'
$cp = "$root\out\classes" + ';C:\elasticsearch-8.17.0\lib\*'

$classes = Join-Path $root 'out\classes'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

$sources = Get-ChildItem -Path (Join-Path $root 'src') -Recurse -Filter *.java | ForEach-Object FullName
Write-Host "== compiling $($sources.Count) source files"
& "$jdk\javac.exe" -encoding UTF-8 -cp 'C:\elasticsearch-8.17.0\lib\*' -d $classes $sources
if ($LASTEXITCODE -ne 0) { throw 'compile failed' }
Write-Host "== compile OK"

$indexDir = 'C:\elasticsearch-8.17.0\data\indices\ny3Uns2iSrCT_x3JOKPukg\0\index'
if (-not (Test-Path $indexDir)) { throw "index dir not found: $indexDir" }

Push-Location 'C:\elasticsearch-8.17.0'
try {
  Write-Host "== phase1 field=dense_vec"
  & "$jdk\java.exe" -Xmx4g "-Dout.dir=$(Join-Path $root 'out')" -cp $cp phase0.Phase1Restart $indexDir dense_vec 300 100 p1-dense 16 500
  if ($LASTEXITCODE -ne 0) { throw 'phase1 dense_vec failed' }

  Write-Host "== phase1 field=content_embedding"
  & "$jdk\java.exe" -Xmx4g "-Dout.dir=$(Join-Path $root 'out')" -cp $cp phase0.Phase1Restart $indexDir content_embedding 300 100 p1-content 16 500
  if ($LASTEXITCODE -ne 0) { throw 'phase1 content_embedding failed' }
} finally {
  Pop-Location
}
