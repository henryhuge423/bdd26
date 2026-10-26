$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Path build_logs -Force | Out-Null
    New-Item -ItemType Directory -Path build -Force | Out-Null
    foreach ($reportName in @('stage_report_6_en', 'stage_report_6_zh', 'summary_plain_zh')) {
        foreach ($passNumber in @(1, 2, 3)) {
            & xelatex -output-directory=build -interaction=nonstopmode -halt-on-error "$reportName.tex" *> "build_logs/$reportName-pass$passNumber.log"
            if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed: $reportName, pass $passNumber" }
        }
        if (Select-String -Path "build/$reportName.log" -Pattern 'Overfull|Missing character|Rerun to get cross-references right' -Quiet) { throw "Layout check failed: $reportName" }
        Copy-Item -LiteralPath "build/$reportName.pdf" -Destination "$reportName.pdf" -Force
    }
} finally { Pop-Location }
