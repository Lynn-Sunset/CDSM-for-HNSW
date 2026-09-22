param(
    [switch]$RegenerateAssets,
    [string]$Python = 'python',
    [string]$PdfLatex = 'pdflatex',
    [string]$BibTeX = 'bibtex'
)
$ErrorActionPreference = 'Stop'
$packageDir = $PSScriptRoot
$paperDir = Join-Path $packageDir 'paper'
$buildDir = Join-Path $packageDir 'tmp/pdfs'
$outputDir = Join-Path $packageDir 'output/pdf'
New-Item -ItemType Directory -Force -Path $buildDir, $outputDir | Out-Null

function Invoke-Checked {
    param([string]$Program, [string[]]$ToolArgs, [string]$LogPath)
    & $Program @ToolArgs 2>&1 | Out-File -LiteralPath $LogPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $LogPath -Tail 25
        throw "$Program failed; see $LogPath"
    }
}

if ($RegenerateAssets) {
    Invoke-Checked $Python @('-B', (Join-Path $packageDir 'build_tables.py')) (Join-Path $buildDir 'tables-build.log')
    $figureDir = Join-Path $paperDir 'figures'
    $figureBuildDir = Join-Path $packageDir 'tmp/figures'
    New-Item -ItemType Directory -Force -Path $figureBuildDir | Out-Null
    Push-Location $figureDir
    try {
        Invoke-Checked $Python @('-B', 'generate_figures.py') (Join-Path $buildDir 'figure-data.log')
        foreach ($figureName in @('method-feedback', 't2i-budget', 'native-tradeoff', 'native-full')) {
            for ($pass = 1; $pass -le 2; $pass++) {
                Invoke-Checked $PdfLatex @('-interaction=nonstopmode', '-halt-on-error', "-output-directory=$figureBuildDir", "$figureName.tex") (Join-Path $buildDir "$figureName-$pass.log")
            }
            Copy-Item -LiteralPath (Join-Path $figureBuildDir "$figureName.pdf") -Destination (Join-Path $figureDir "$figureName.pdf") -Force
        }
    } finally { Pop-Location }
}

Push-Location $paperDir
try {
    $latexArgs = @('-interaction=nonstopmode', '-halt-on-error', "-output-directory=$buildDir", 'main.tex')
    Invoke-Checked $PdfLatex $latexArgs (Join-Path $buildDir 'compile-1.log')
    Invoke-Checked $BibTeX @('../tmp/pdfs/main') (Join-Path $buildDir 'bibtex-build.log')
    Invoke-Checked $PdfLatex $latexArgs (Join-Path $buildDir 'compile-2.log')
    Invoke-Checked $PdfLatex $latexArgs (Join-Path $buildDir 'compile-3.log')
} finally { Pop-Location }

$builtPdf = Join-Path $buildDir 'main.pdf'
if (Get-Command pdfinfo -ErrorAction SilentlyContinue) {
    $pdfMetadata = & pdfinfo $builtPdf
    if ($LASTEXITCODE -ne 0) { throw 'pdfinfo failed.' }
    $pageLine = $pdfMetadata | Where-Object { $_ -match '^Pages:\s+\d+' }
    if ($pageLine -match '^Pages:\s+(\d+)') {
        $pageCount = [int]$Matches[1]
        if ($pageCount -gt 12) { throw "PDF has $pageCount pages; the limit is 12." }
        Write-Host "PDF pages (including appendix and references): $pageCount"
    }
}
$finalPdf = Join-Path $outputDir 'cdsm-paper.pdf'
Copy-Item -LiteralPath $builtPdf -Destination $finalPdf -Force
Write-Host "Built $finalPdf"
Write-Host 'QA.json records the delivered edition; re-render and inspect after subsequent edits.'
