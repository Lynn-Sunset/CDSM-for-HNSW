# T2I 结果后分析：平台-增益回归 + 分桶表
# 用法：在 phaseT2I-*.csv 生成后运行
param([string]$Tag = 'pT2I-2000')

$csv = Import-Csv "C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\phaseT2I-$Tag.csv"
$n = $csv.Count
if ($n -eq 0) { "CSV not found or empty"; exit 1 }

"===== T2I $Tag (n=$n) ====="
"base@100={0:F3}  single@1600={1:F3}  marked4={2:F3}" -f `
  ($csv | Measure-Object baseRec -Average).Average, `
  ($csv | Measure-Object single1600 -Average).Average, `
  ($csv | Measure-Object marked4 -Average).Average

# 难查询子集
$hard = $csv | Where-Object { [double]$_.baseRec -lt 0.9 }
if ($hard.Count -gt 0) {
  "--- hard (base<0.9, n=$($hard.Count)) ---"
  foreach ($c in 'random4','dirAware4','farthest4','marked4') {
    $g = @($hard | ForEach-Object { [double]$_.$c - [double]$_.single1600 }) | Measure-Object -Average
    $br = @($hard | Where-Object { [double]$_.$c -gt [double]$_.single1600 + 0.01 }).Count
    "{0,-11}: recall={1:F3}  gain={2:+0.000;-0.000}  breaks {3}/{4} ({5:F1}%)" -f $c, `
      ($hard | Measure-Object $c -Average).Average, $g.Average, $br, $hard.Count, (100.0*$br/$hard.Count)
  }
}

# 平台-增益回归：gain = alpha*(1-P) + beta
$xs = @($csv | ForEach-Object { 1 - [double]$_.single1600 })
$ys = @($csv | ForEach-Object { [double]$_.marked4 - [double]$_.single1600 })
$mx = ($xs | Measure-Object -Average).Average
$my = ($ys | Measure-Object -Average).Average
$sxy = 0.0; $sxx = 0.0
for ($i = 0; $i -lt $n; $i++) { $dx = $xs[$i] - $mx; $sxy += $dx * ($ys[$i] - $my); $sxx += $dx * $dx }
if ($sxx -gt 0) { "regression: gain(marked) = {0:F3}*(1-P) + {1:F3}" -f ($sxy/$sxx), ($my - ($sxy/$sxx)*$mx) }

# 分桶（按单次@1600 平台高度）
"--- bucket by single1600 ---"
$csv | Group-Object { $p=[double]$_.single1600; if ($p -lt 0.5) {'[0,0.5)'} elseif ($p -lt 0.7) {'[0.5,0.7)'} elseif ($p -lt 0.9) {'[0.7,0.9)'} else {'[0.9,1.0]'} } | Sort-Object Name | ForEach-Object {
  $g = @($_.Group | ForEach-Object { [double]$_.marked4 - [double]$_.single1600 }) | Measure-Object -Average
  $br = @($_.Group | Where-Object { [double]$_.marked4 -gt [double]$_.single1600 + 0.01 }).Count
  "{0,-9} n={1,-5} markedGain={2:+0.000;-0.000}  breaks {3}/{4} ({5:F1}%)" -f $_.Name, $_.Count, $g.Average, $br, $_.Count, (100.0*$br/$_.Count)
}
