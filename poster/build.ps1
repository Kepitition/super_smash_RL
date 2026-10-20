# Builds poster.pdf (A2) with pdflatex + bibtex (MiKTeX or TeX Live on the PATH).
#   powershell -ExecutionPolicy Bypass -File poster\build.ps1
# Regenerate the figures and numbers first with: python poster\make_figures.py
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    pdflatex -interaction=nonstopmode poster.tex | Out-Null
    bibtex poster | Out-Null
    pdflatex -interaction=nonstopmode poster.tex | Out-Null
    pdflatex -interaction=nonstopmode poster.tex | Select-String -Pattern "^!|Output written|Overfull"
} finally {
    Pop-Location
}
