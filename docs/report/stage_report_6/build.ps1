$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Path build_logs -Force | Out-Null
    foreach ($reportName in @('stage_report_6_en', 'stage_report_6_zh', 'summary_plain_zh')) {
        foreach ($passNumber in @(1, 2)) {
            & xelatex -interaction=nonstopmode -halt-on-error "$reportName.tex" *> "build_logs/$reportName-pass$passNumber.log"
            if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed: $reportName, pass $passNumber" }
        }
        if (Select-String -Path "$reportName.log" -Pattern 'Overfull|Missing character' -Quiet) { throw "Layout check failed: $reportName" }
    }
} finally { Pop-Location }
