# Phase 0 real-data run on chunks-cx-paper-v1 (413k docs, 33.2GB)
# Two runs: dense_vec and content_embedding (both 2048-d cosine HNSW).

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

# chunks-cx-paper-v1 shard 0 (uuid confirmed via _settings API)
$indexDir = 'C:\elasticsearch-8.17.0\data\indices\ny3Uns2iSrCT_x3JOKPukg\0\index'
if (-not (Test-Path $indexDir)) { throw "index dir not found: $indexDir" }

# run with ES home as working directory (native lib resolution)
Push-Location 'C:\elasticsearch-8.17.0'
try {
  Write-Host "== running field=dense_vec"
  & "$jdk\java.exe" -Xmx4g "-Dout.dir=$(Join-Path $root 'out')" -cp $cp phase0.Phase0Real $indexDir dense_vec 300 100 paper-dense `
      (Join-Path $root 'out\es-baseline-query-dense_vec.txt') (Join-Path $root 'out\es-baseline-top10-dense_vec.txt')
  if ($LASTEXITCODE -ne 0) { throw 'run dense_vec failed' }

  Write-Host "== running field=content_embedding"
  & "$jdk\java.exe" -Xmx4g "-Dout.dir=$(Join-Path $root 'out')" -cp $cp phase0.Phase0Real $indexDir content_embedding 300 100 paper-content `
      (Join-Path $root 'out\es-baseline-query-content_embedding.txt') (Join-Path $root 'out\es-baseline-top10-content_embedding.txt')
  if ($LASTEXITCODE -ne 0) { throw 'run content_embedding failed' }
} finally {
  Pop-Location
}
