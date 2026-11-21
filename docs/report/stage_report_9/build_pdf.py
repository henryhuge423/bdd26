"""Compile the three self-contained LaTeX reports with XeLaTeX."""
from pathlib import Path
import subprocess,sys

root=Path(__file__).resolve().parent
for name in ('stage_report_9_zh','stage_report_9_en','summary_plain_zh'):
 for pass_number in (1,2):
  result=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=root,capture_output=True)
  (root/(name+f'-pass{pass_number}.log')).write_bytes(result.stdout+result.stderr)
  if result.returncode:
   print(result.stdout.decode(errors='replace')[-3000:]);sys.exit(result.returncode)
print('Compiled all three reports from self-contained LaTeX sources.')
