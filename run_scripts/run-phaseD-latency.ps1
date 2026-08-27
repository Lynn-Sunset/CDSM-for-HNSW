# Phase D wall-clock latency benchmark on chunks-cx-paper-v1 (dense_vec, 2048-d cosine HNSW).
# Methodology: fixed 500-query set, 2 warmup passes, 7 measurement rounds with
# shuffled method order, per-query median, p50/p90/p99/max reporting.

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

# run with ES home as working directory (native lib resolution);
# fixed heap (-Xms=-Xmx) + AlwaysPreTouch to keep GC/page-fault noise out of timing
Push-Location 'C:\elasticsearch-8.17.0'
try {
  Write-Host "== running PhaseDLatency dense_vec 500 queries"
  & "$jdk\java.exe" -Xms8g -Xmx8g -XX:+AlwaysPreTouch "-Dout.dir=$(Join-Path $root 'out')" `
      -cp $cp phase0.PhaseDLatency $indexDir dense_vec 500 lat-dense
  if ($LASTEXITCODE -ne 0) { throw 'PhaseDLatency failed' }
} finally {
  Pop-Location
}
