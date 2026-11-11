"""Task-owned queue; never terminates a process it did not launch."""
import os,sys,json,time,subprocess,datetime,hashlib
from pathlib import Path
TASK=Path(__file__).resolve().parent
PLAN=TASK/'plan.json'
def utc():return datetime.datetime.now(datetime.timezone.utc)
def log(event,**kw):
 row={'utc':utc().isoformat(),'event':event,**kw}
 with (TASK/'logs/scheduler.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blocked():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 mapping={}; busy=set()
 for row in rows:
  i,uid,mem,util=[z.strip() for z in row.split(',')];i=int(i);mapping[uid]=i
  if int(mem)>1024 or int(util)>10:busy.add(i)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,process_name','--format=csv,noheader'],text=True).splitlines()
 for row in apps:
  if ',' not in row:continue
  uid,name=row.split(',',1)
  if 'python' in name.lower() and uid.strip() in mapping:busy.add(mapping[uid.strip()])
 return busy
def main():
 # Exclusive task lock protects against duplicate scheduling. No stale-lock auto-removal.
 lock=TASK/'scheduler.lock'
 with lock.open('x',encoding='utf-8') as f:f.write(json.dumps({'pid':os.getpid(),'utc':utc().isoformat()}))
 running={};terminal=set()
 ledger=TASK/'logs/scheduler.jsonl'
 if ledger.exists():
  for line in ledger.read_text(encoding='utf-8').splitlines():
   row=json.loads(line)
   if row['event'] in ['completed','failed','timeout','budget_terminated']:terminal.add(row['job'])
 log('scheduler_started',pid=os.getpid())
 try:
  while True:
   plan=json.loads(PLAN.read_text(encoding='utf-8'))
   deadline=datetime.datetime.fromisoformat(plan['deadline_utc'])
   expired=utc()>=deadline
   for gpu,item in list(running.items()):
    proc,job,start,out,err=item
    reason=None
    if proc.poll() is None:
     if expired:reason='budget_terminated'
     elif time.monotonic()-start>plan['per_fold_timeout_seconds']:reason='timeout'
     if reason:
      proc.terminate()
      try:proc.wait(timeout=30)
      except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=30)
    if proc.poll() is not None:
     out.close();err.close()
     c=job['config'];result=TASK/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))/'result.json'
     ok=proc.returncode==0 and result.exists()
     if ok:
      value=json.loads(result.read_text(encoding='utf-8'))
      ok=value['config']['experiment']==c['experiment'] and value['config']['fold']==c['fold']
     log(reason or ('completed' if ok else 'failed'),job=job['id'],pid=proc.pid,gpu=gpu,exit_code=proc.returncode,result=str(result),seconds=time.monotonic()-start)
     terminal.add(job['id']);del running[gpu]
   if expired:
    log('budget_finished',completed_or_failed=len(terminal));break
   active={v[1]['id'] for v in running.values()}
   pending=[j for j in plan['jobs'] if j['id'] not in terminal|active]
   if not pending and not running:
    log('queue_finished',completed_or_failed=len(terminal));break
   try:busy=blocked()
   except Exception as e:
    log('resource_query_failed',error=str(e));time.sleep(20);continue
   for gpu in plan['gpu_ids']:
    if not pending or gpu in running or gpu in busy:continue
    # Freeze all executable code before launch; any mutation blocks further launches.
    expected=plan['execution_sha256']
    assert all(digest(TASK/p)==h for p,h in expected.items()),'Execution code changed after plan freeze'
    job=pending.pop(0)
    c=job['config'];dest=TASK/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))
    if dest.exists():
     log('failed',job=job['id'],error='Output already exists; refusing overwrite');terminal.add(job['id']);continue
    out=(TASK/'logs'/(job['id']+'.out')).open('x',encoding='utf-8');err=(TASK/'logs'/(job['id']+'.err')).open('x',encoding='utf-8')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',DGLBACKEND='pytorch',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    proc=subprocess.Popen([sys.executable,'-B',str(TASK/'worker.py'),'--job',job['id'],'--gpu',str(gpu)],cwd=str(TASK/'code'),stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    running[gpu]=(proc,job,time.monotonic(),out,err)
    log('started',job=job['id'],gpu=gpu,pid=proc.pid,plan_sha256=digest(PLAN))
   time.sleep(20)
 finally:
  # A scheduler error must not orphan the workers it owns.
  for gpu,(proc,job,start,out,err) in running.items():
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=30)
    log('failed',job=job['id'],pid=proc.pid,error='Scheduler failed; owned worker terminated')
   out.close();err.close()
  lock.unlink()
if __name__=='__main__':main()
