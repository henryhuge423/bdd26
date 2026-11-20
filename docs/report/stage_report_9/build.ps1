$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
  foreach ($name in @('stage_report_9_en','stage_report_9_zh','summary_plain_zh')) {
    foreach ($pass in 1..2) {
      & xelatex -interaction=nonstopmode -halt-on-error -file-line-error "$name.tex"
      if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed: $name pass $pass" }
    }
  }
} finally {Pop-Location}
