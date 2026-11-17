#!/bin/bash
#SBATCH --job-name=bdd26-p31
#SBATCH --account=jhu159
#SBATCH --partition=gpu-shared
#SBATCH --qos=gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --constraint=lustre
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --export=NONE
set -euo pipefail
trap 'echo "phase_exit utc=$(date -u +%FT%TZ) rc=$?"' EXIT
echo "phase_start utc=$(date -u +%FT%TZ)"
umask 077
base=/expanse/lustre/projects/jhu159/xjin10/bdd26
source="$base/p31_20261116_01/snapshot"
root=/expanse/lustre/scratch/xjin10/temp_project/bdd26/p31_20261116_01
python="$base/runtimes/p3_20261020_01/bin/python"
data="$base/migration_20261020/data/pannuke"
weights="$base/migration_20261020/weights/MahmoodLab/UNI/pytorch_model.bin"
quota="$source/scripts/check_lustre_quota.py"
arms=(C G C G)
seeds=(19 19 1 1)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
tag="A_${arm}_s${seed}"
cd "$root"
export SLURM_EXPORT_ENV=ALL
export PYTHONHOME=/expanse/lustre/projects/jhu159/xjin10/bdd26/runtime_startup_p30_20261115_01
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/expanse/lustre/projects/jhu159/xjin10/bdd26/runtimes/p3_20261020_01/lib
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="/scratch/$USER/job_$SLURM_JOB_ID"
export XDG_CACHE_HOME="$TMPDIR/cache" HF_HOME="$TMPDIR/hf" TORCH_HOME="$TMPDIR/torch"
test -d "$TMPDIR"
export PYTHONPYCACHEPREFIX="$TMPDIR/no_existing_bytecode"
export PYTHONPATH="$source/src:$source/scripts:/expanse/lustre/projects/jhu159/xjin10/bdd26/runtime_native_p30_20261115_01/lib/python3.10/lib-dynload:$TMPDIR/p30_site_packages:/expanse/lustre/projects/jhu159/xjin10/bdd26/runtimes/p3_20261020_01/lib/python3.10/site-packages"
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u PYTHONPYCACHEPREFIX python3 - "$TMPDIR/p30_site_packages" <<'PACKAGE_BOOTSTRAP'
"""Assemble byte-identical Python sources and small Torch natives on node-local storage."""
from pathlib import Path,PurePosixPath
import sys,json,hashlib,tarfile,os
base=Path('/expanse/lustre/projects/jhu159/xjin10/bdd26')
stage=base/'runtime_packages_p30_20261115_02';native=base/'runtime_native_torch_p30_20261115_01'
original=base/'runtimes/p3_20261020_01/lib/python3.10/site-packages'
target=Path(sys.argv[1]);assert not target.exists();target.mkdir()
pm=stage/'package_manifest.json';assert hashlib.sha256(pm.read_bytes()).hexdigest()=='fca061562aafacfbe2a2b6150f7484427a7673d9d41c73219412c1c238c51cca'
source=json.loads(pm.read_text())
nm=native/'native_torch_manifest.json';assert hashlib.sha256(nm.read_bytes()).hexdigest()=='b92fe29c73d34fc6b191510beea51b87c5a17ced36f3f81948ee003e47107685'
natives=json.loads(nm.read_text());assert len(natives['files'])==7
numpy_stage=base/'runtime_native_numpy_p30_20261115_01'
numpy_mb=(numpy_stage/'native_numpy_manifest.json').read_bytes();assert hashlib.sha256(numpy_mb).hexdigest()=='35d30b7f00af09e238fdbfc8247b0c851a1525a259b096e5ee8714eeadd98419'
numpy_native=json.loads(numpy_mb);assert len(numpy_native['files'])==9 and numpy_native['original_RECORD_matching']
exact_natives={**natives['files'],**numpy_native['files']}

for archive,expected,mapping in ((stage/'python_sources.tar.xz',source['bundle_sha256'],source['files']),(native/'native_torch.tar.xz',natives['archive_sha256'],natives['files']),(numpy_stage/'native_numpy.tar.xz',numpy_native['archive_sha256'],numpy_native['files'])):
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
 with tarfile.open(archive,'r:xz') as tar:
  seen=set()
  for member in tar:
   pure=PurePosixPath(member.name);assert member.isfile() and not pure.is_absolute() and '..' not in pure.parts and member.name in mapping and member.name not in seen
   seen.add(member.name);data=tar.extractfile(member).read();e=mapping[member.name];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
   p=target/member.name;p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(data)
  assert seen==set(mapping)
for name,rel in source['links'].items():
 pure=PurePosixPath(name);assert not pure.is_absolute() and '..' not in pure.parts
 p=target/name
 if name in exact_natives:
  assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==exact_natives[name]['sha256']
 elif name=='torch/lib':
  assert p.is_dir() and not p.is_symlink()
  for child in (original/rel).iterdir():
   n=name+'/'+child.name;t=target/n
   if n in natives['files']:assert t.is_file() and not t.is_symlink()
   else:assert not t.exists() and not t.is_symlink();t.symlink_to(child)
 else:
  assert not p.exists() and not p.is_symlink(),name;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(original/rel)
adjacent=[]
for name in ('Pillow.libs','numpy.libs','scipy.libs','torchvision.libs','nvidia'):
 reference=original/name;p=target/name
 if reference.is_dir():
  if p.exists() or p.is_symlink():assert p.is_symlink() and p.resolve()==reference.resolve(),name
  else:p.symlink_to(reference)
  adjacent.append(name)
assert 'Pillow.libs' in adjacent
print('PINNED_ADJACENT_LIBRARY_NAMESPACES_PASS',adjacent,flush=True)
platform_source=base/'runtime_stdlib_patch_p30_20261115_01/platform.py'
platform_data=platform_source.read_bytes();assert hashlib.sha256(platform_data).hexdigest()=='87e7d2f54b21b290b35dfa2bdcb66e4d80a275be8c9b6abd0fae3ceb3dd62547'
with (target/'platform.py').open('xb') as f:f.write(platform_data)
print('ORIGINAL_CONDA_FORGE_PLATFORM_PASS',len(platform_data),flush=True)
print('EXACT_NUMPY_RANDOM_NATIVES_PASS',len(numpy_native['files']),flush=True)
print('NODE_LOCAL_PYTHON_AND_TORCH_NATIVES_PASS',len(source['files']),len(natives['files']),flush=True)
PACKAGE_BOOTSTRAP
export LD_LIBRARY_PATH="$TMPDIR/p30_site_packages/torch/lib:$base/runtimes/p3_20261020_01/lib"

test ! -e "$root/$tag"
df -h "$root" "$base" "$TMPDIR"
"$python" "$quota" --filesystem /expanse/lustre/scratch --scope user --identity "$USER" --min-free-gib 80 > "$root/$tag.quota_scratch_user.jsonl"
"$python" "$quota" --filesystem /expanse/lustre/projects --scope user --identity "$USER" --min-free-gib 80 > "$root/$tag.quota_project_user.jsonl"
"$python" "$quota" --filesystem /expanse/lustre/projects --scope group --identity jhu159 --min-free-gib 80 > "$root/$tag.quota_project_group.jsonl"
echo "phase_quota_pass utc=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=uuid,name,memory.free,memory.used --format=csv
"$python" - "$source" "$root" "$tag" <<'PY'
import hashlib,json,os,pathlib,shutil,socket,sys,torch
source,root=map(pathlib.Path,sys.argv[1:3]);tag=sys.argv[3]

startup=pathlib.Path('/expanse/lustre/projects/jhu159/xjin10/bdd26/runtime_startup_p30_20261115_01')
assert sys.prefix==str(startup)
assert hashlib.sha256((startup/'source_manifest.json').read_bytes()).hexdigest()=='0f8d8390c1ccc0d9a422c6a12f822230519ee19afc8e1e5b2d1ad878b8a967bc'
startup_manifest=json.loads((startup/'source_manifest.json').read_text())
assert startup_manifest['archive_sha256']=='2ba22d68f6bfb0ffcfb3729ec382d5905fdebdd56b82a4ad46add9e65e95db22' and len(startup_manifest['hashes'])==870
for rel,expected in startup_manifest['hashes'].items():assert hashlib.sha256((startup/'lib/python3.10'/rel).read_bytes()).hexdigest()==expected,rel


native=pathlib.Path('/expanse/lustre/projects/jhu159/xjin10/bdd26/runtime_native_p30_20261115_01');package=pathlib.Path('/expanse/lustre/projects/jhu159/xjin10/bdd26/runtime_packages_p30_20261115_02')
assert hashlib.sha256((native/'native_manifest.json').read_bytes()).hexdigest()=='8322180570c00b5262c4337b60c83eb2db1fad68caeec0e53adbea23f33cb56f'
assert hashlib.sha256((package/'package_manifest.json').read_bytes()).hexdigest()=='fca061562aafacfbe2a2b6150f7484427a7673d9d41c73219412c1c238c51cca'
for rel,e in json.loads((native/'native_manifest.json').read_text())['files'].items():assert hashlib.sha256((native/rel).read_bytes()).hexdigest()==e['sha256']
assert hashlib.sha256((package/'python_sources.tar.xz').read_bytes()).hexdigest()==json.loads((package/'package_manifest.json').read_text())['bundle_sha256']

import subprocess
layout=subprocess.check_output(['lfs','getstripe','-d','-i',str(root)],universal_newlines=True,timeout=15);assert int(layout.strip())==43
space=subprocess.check_output(['lfs','df','/expanse/lustre/projects'],universal_newlines=True,timeout=30)
assert '1K-blocks' in space.splitlines()[0]
rows=[v for v in space.splitlines() if '[OST:43]' in v];assert len(rows)==1
assert int(rows[0].split()[3])*1024>=150*1024**3

ready=pathlib.Path('/expanse/lustre/scratch/xjin10/temp_project/bdd26/p31_smoke_20261116_01/ready.json')
readiness=json.loads(ready.read_text());assert readiness['status']=='PASS' and all(r['safety_seconds']<43200 for r in readiness['reports'].values())
sys.path.insert(0,str(source/'scripts'))
from p5_contracts import COHORT_SHA,TISSUE_SHA,verify_hashes
assert readiness['identity']['cohort_sha256']==COHORT_SHA
assert readiness['identity']['input_sha256']['fold1/tissue.npy']==TISSUE_SHA
verify_hashes(source.parent,readiness['script_hashes'])
assert readiness['source_manifest_sha256']==hashlib.sha256((source.parent/'source_hashes.json').read_bytes()).hexdigest()
for rel,expected in json.loads((source.parent/'source_hashes.json').read_text()).items():
 assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==expected,rel
assert torch.__version__=='2.5.1+cu124' and torch.cuda.device_count()==1
free,total=torch.cuda.mem_get_info();assert free>=28*1024**3
assert shutil.disk_usage(root).free>=80*1024**3
record=dict(job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],task=os.environ['SLURM_ARRAY_TASK_ID'],hostname=socket.gethostname(),gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],free_memory=free,torch=torch.__version__,precision='original BF16 autocast / FP32 parameters and loss',no_test_evaluation=True)
with (root/(tag+'.launch.json')).open('x') as f:json.dump(record,f,indent=2)
PY
echo "phase_identity_pass utc=$(date -u +%FT%TZ)"
micro=4
srun "$python" -u "$source/scripts/train_proposal_p31.py" --data "$data" --weights "$weights" --out "$root/$tag" --arm "$arm" --seed "$seed" --microbatch "$micro" > "$root/$tag.log" 2>&1
"$python" - "$root/$tag" "$base" "$quota" <<'PY'
import hashlib,json,pathlib,shutil,subprocess,sys
run,base=map(pathlib.Path,sys.argv[1:3]);quota=sys.argv[3]

native=pathlib.Path('/expanse/lustre/projects/jhu159/xjin10/bdd26/runtime_native_p30_20261115_01');package=pathlib.Path('/expanse/lustre/projects/jhu159/xjin10/bdd26/runtime_packages_p30_20261115_02')
assert hashlib.sha256((native/'native_manifest.json').read_bytes()).hexdigest()=='8322180570c00b5262c4337b60c83eb2db1fad68caeec0e53adbea23f33cb56f'
assert hashlib.sha256((package/'package_manifest.json').read_bytes()).hexdigest()=='fca061562aafacfbe2a2b6150f7484427a7673d9d41c73219412c1c238c51cca'
for rel,e in json.loads((native/'native_manifest.json').read_text())['files'].items():assert hashlib.sha256((native/rel).read_bytes()).hexdigest()==e['sha256']
assert hashlib.sha256((package/'python_sources.tar.xz').read_bytes()).hexdigest()==json.loads((package/'package_manifest.json').read_text())['bundle_sha256']

import subprocess
layout=subprocess.check_output(['lfs','getstripe','-d','-i',str(base/'p31_20261116_01/results')],universal_newlines=True,timeout=15);assert int(layout.strip())==43
space=subprocess.check_output(['lfs','df','/expanse/lustre/projects'],universal_newlines=True,timeout=30)
assert '1K-blocks' in space.splitlines()[0]
rows=[v for v in space.splitlines() if '[OST:43]' in v];assert len(rows)==1
assert int(rows[0].split()[3])*1024>=150*1024**3

size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
required=max(80*1024**3,int(size*1.1)+20*1024**3)
assert shutil.disk_usage(base).free>=required,'project filesystem headroom'
for scope,identity in [('user','xjin10'),('group','jhu159')]:
 subprocess.check_call([sys.executable,quota,'--filesystem','/expanse/lustre/projects','--scope',scope,'--identity',identity,'--min-free-gib',str(required/1024**3)])
PY
echo "phase_preserve_start utc=$(date -u +%FT%TZ)"
result="$base/p31_20261116_01/results/$tag"
test ! -e "$result"
mkdir "$result"
cp -a "$root/$tag/." "$result/"
cp "$root/$tag.log" "$root/$tag.launch.json" "$result/"
"$python" - "$result" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);assert json.loads((root/'complete.json').read_text())['epochs']==130
hashes={}
for p in sorted(root.rglob('*')):
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  hashes[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
with (root/'preservation_manifest.json').open('x') as f:json.dump(hashes,f,indent=2)
PY
