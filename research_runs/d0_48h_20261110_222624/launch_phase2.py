import sys,runpy,traceback
from pathlib import Path
T=Path(__file__).resolve().parent
sys.stdout=(T/'logs/phase2.out').open('x',encoding='utf-8',buffering=1)
sys.stderr=(T/'logs/phase2.err').open('x',encoding='utf-8',buffering=1)
try:runpy.run_path(str(T/'scheduler.py'),run_name='__main__')
except BaseException:traceback.print_exc();raise
