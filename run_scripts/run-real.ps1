# Phase 0 real-data run script
# Runs the instrumented search over the ACTUAL production HNSW graphs read from
# the local Elasticsearch 8.17.0 index segments (no rebuild), and cross-validates
# against the ES API baseline saved in out/es-baseline-*.txt.
#
# Paths: override $env:ES817_HOME for a different installation; pass -indexDir to
# point at a different index (default: the medical chunks-drug-label-v3 shard 0
# whose uuid was confirmed via the ES API). External-review item 3: no hardcoded paths.

param([string]$indexDir = '')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$esHome = if ($env:ES817_HOME) { $env:ES817_HOME } else { 'C:\elasticsearch-8.17.0' }
$jdk = Join-Path $esHome 'jdk\bin'
# full ES lib on classpath: the index uses ES-specific codecs/postings formats
# (ES87BloomFilter etc.) whose SPI classes live in elasticsearch-*.jar
$cp = "$root\out\classes" + ';' + (Join-Path $esHome 'lib\*')

$classes = Join-Path $root 'out\classes'
New-Item -ItemType Directory -Force -Path $classes | Out-Null

if (-not $indexDir) {
  # chunks-drug-label-v3 shard 0 (index uuid confirmed via ES API error message)
  $indexDir = Join-Path $esHome 'data\indices\Yl_8DXT3Q9ys1cKd-yEHDA\0\index'
}

$sources = Get-ChildItem -Path (Join-Path $root 'src') -Recurse -Filter *.java | ForEach-Object FullName
Write-Host "== compiling $($sources.Count) source files"
& "$jdk\javac.exe" -encoding UTF-8 -cp (Join-Path $esHome 'lib\*') -d $classes $sources
if ($LASTEXITCODE -ne 0) { throw 'compile failed' }
Write-Host "== compile OK"

if (-not (Test-Path $indexDir)) { throw "index dir not found: $indexDir (pass -indexDir)" }

# run with ES home as working directory: the ES nativeaccess loader resolves
# lib\platform\windows-x64\*.dll relative to the working directory
Push-Location $esHome
try {
  & "$jdk\java.exe" -Xmx4g "-Dout.dir=$(Join-Path $root 'out')" -cp $cp phase0.Phase0Real $indexDir dense_vec 300 100 druglabel `
      (Join-Path $root 'out\es-baseline-query.txt') (Join-Path $root 'out\es-baseline-top10.txt')
  if ($LASTEXITCODE -ne 0) { throw 'Phase0Real run failed' }
} finally {
  Pop-Location
}
