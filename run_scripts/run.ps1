# Phase 0 build & run script
# Compiles against the local Elasticsearch 8.17.0 bundled Lucene (lucene-core-9.12.0.jar)
# and runs three experiment configs:
#   A: well-separated clusters (sigma=0.25, visitLimit=300)
#   A0: same data, unlimited visit budget (reference)
#   B: heavily overlapping clusters (sigma=0.90, visitLimit=300)
#
# Paths: override $env:ES817_HOME to point at a different ES 8.17 installation;
# the default below is this machine's location (external-review item 3: no hardcoded paths).

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$esHome = if ($env:ES817_HOME) { $env:ES817_HOME } else { 'C:\elasticsearch-8.17.0' }
$jdk = Join-Path $esHome 'jdk\bin'
$lucene = Join-Path $esHome 'lib\lucene-core-9.12.0.jar'

if (-not (Test-Path $lucene)) { throw "Lucene jar not found: $lucene (set `$env:ES817_HOME)" }

$classes = Join-Path $root 'out\classes'
$outDir = Join-Path $root 'out'
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$sources = Get-ChildItem -Path (Join-Path $root 'src') -Recurse -Filter *.java | ForEach-Object FullName
Write-Host "== compiling $($sources.Count) source files"
& "$jdk\javac.exe" -encoding UTF-8 -cp $lucene -d $classes $sources
if ($LASTEXITCODE -ne 0) { throw 'compile failed' }
Write-Host "== compile OK"

& "$jdk\java.exe" -cp "$classes;$lucene" phase0.Phase0Main 0.25 A 300
if ($LASTEXITCODE -ne 0) { throw 'run A failed' }
& "$jdk\java.exe" -cp "$classes;$lucene" phase0.Phase0Main 0.25 A0 0
if ($LASTEXITCODE -ne 0) { throw 'run A0 failed' }
& "$jdk\java.exe" -cp "$classes;$lucene" phase0.Phase0Main 0.90 B 300
if ($LASTEXITCODE -ne 0) { throw 'run B failed' }

Write-Host "== done"
