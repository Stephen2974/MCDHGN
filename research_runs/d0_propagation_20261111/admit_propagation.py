"""Wait on the exact current scheduler process handle, then validate admission."""
import sys,json,datetime,time,ctypes,hashlib,runpy,traceback
from pathlib import Path
T=Path(__file__).resolve().parent
S=T.parent/'d0_48h_20261110_222624'
sys.stdout=(T/'logs/admission.out').open('x',encoding='utf-8',buffering=1)
sys.stderr=(T/'logs/admission.err').open('x',encoding='utf-8',buffering=1)
def emit(event,**kw):
 print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw)),flush=True)
def main():
 deadline=datetime.datetime.fromisoformat('2026-11-12T14:26:23+00:00')
 kernel=ctypes.WinDLL('kernel32',use_last_error=True)
 kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
 kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
 kernel.CloseHandle.argtypes=[ctypes.c_void_p]
 handle=kernel.OpenProcess(0x100000,False,9680)
 if not handle:raise RuntimeError('Expected scheduler PID unavailable; manual revalidation required')
 emit('waiting_on_scheduler',pid=9680)
 try:
  while kernel.WaitForSingleObject(handle,30000)==258:
   if datetime.datetime.now(datetime.timezone.utc)>=deadline:emit('deadline_before_admission');return
 finally:kernel.CloseHandle(handle)
 if datetime.datetime.now(datetime.timezone.utc)>=deadline:emit('deadline_before_admission');return
 events=[json.loads(x) for x in (S/'logs/scheduler.jsonl').read_text(encoding='utf-8').splitlines()]
 assert events[-1]['event']=='queue_finished','Prior scheduler did not finish normally'
 old=json.loads((S/'plan.json').read_text(encoding='utf-8'))
 assert len(old['jobs'])==140,'Prior plan changed; review needed'
 for job in old['jobs']:
  c=job['config'];p=S/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))/'result.json'
  assert p.exists(),str(p)
 assert not (S/'scheduler.lock').exists(),'Prior lock still present'
 plan=json.loads((T/'plan.json').read_text(encoding='utf-8'))
 assert datetime.datetime.fromisoformat(plan['deadline_utc'])==deadline
 assert all(hashlib.sha256((T/p).read_bytes()).hexdigest()==h for p,h in plan['execution_sha256'].items())
 assert len(list((T/'outputs/smoke').glob('*/result.json')))==5
 emit('admission_passed',jobs=len(plan['jobs']),deadline=deadline.isoformat())
 runpy.run_path(str(T/'scheduler.py'),run_name='__main__')
try:main()
except BaseException:traceback.print_exc();raise
