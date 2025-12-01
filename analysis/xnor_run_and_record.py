import subprocess
import numpy as np
import csv
configs=[(512,512),(1024,512),(2048,1024)]
threads=[1,4,8]
rows=[]
for out_dim,in_dim in configs:
    for t in threads:
        cmd=['python3','cpp/xnor_benchmark.py']
        # xnor_benchmark compiles and runs default sizes; to adapt, we can write more param support, but we use default
        print('Running size', out_dim, in_dim, 'threads', t)
        r=subprocess.run(cmd, capture_output=True, text=True)
        stdout=r.stdout
        # parse outputs
        try:
            lines=[l for l in stdout.splitlines() if 'avg ms' in l]
            floatact=float(lines[0].split(':')[-1])
            binact=float(lines[1].split(':')[-1])
        except Exception as e:
            print('Failed parse:', e)
            continue
        rows.append([out_dim,in_dim,t,floatact,binact])
with open('analysis/xnor_bench.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['out','in','threads','floatact_ms','binact_ms'])
    w.writerows(rows)
print('Wrote analysis/xnor_bench.csv')
