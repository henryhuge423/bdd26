"""Use the sealed quota checker before spending a once-only SLURM submission."""
import json
import subprocess


def admit_storage(python, helper, run=subprocess.run):
    records={}
    for name,filesystem,scope,identity in (
        ('scratch_user','/expanse/lustre/scratch','user','xjin10'),
        ('project_user','/expanse/lustre/projects','user','xjin10'),
        ('project_group','/expanse/lustre/projects','group','jhu159')):
        cmd=[str(python),str(helper),'--filesystem',filesystem,'--scope',scope,'--identity',identity,
             '--min-free-gib','100','--min-free-files','1000']
        result=run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=600)
        rows=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        if result.returncode!=0 or not rows or rows[-1].get('status')!='PASS':
            raise RuntimeError('storage admission denied for '+name+': '+result.stdout+'\n'+result.stderr)
        records[name]=rows
    return records
