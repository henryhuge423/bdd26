$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
 New-Item -ItemType Directory -Force build,build_logs | Out-Null
 foreach ($name in @('stage_report_7_en','stage_report_7_zh','summary_plain_zh')) {
  foreach ($pass in 1..3) {
   & xelatex -output-directory=build -interaction=nonstopmode -halt-on-error "$name.tex" *> "build_logs/$name-pass$pass.log"
   if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed: $name pass $pass" }
  }
  if (Select-String -Path "build/$name.log" -Pattern 'Overfull|Missing character|Rerun to get cross-references right' -Quiet) { throw "Layout warning: $name" }
 }
} finally { Pop-Location }
