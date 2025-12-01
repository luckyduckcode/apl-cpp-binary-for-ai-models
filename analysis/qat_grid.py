import subprocess
import os
import csv
import numpy as np
import time

def run_qat(epochs=10, lr=1e-3, batch_size=16, learnable=False, prefix='qat'):
    cmd = ['python3', 'distillation.py', '--qat', '--epochs', str(epochs), '--lr', str(lr), '--batch_size', str(batch_size)]
    if learnable:
        cmd.append('--learnable_scale')
    subprocess.run(cmd, check=True)
    # After training, the generated files are student_fp32.pth and student_quantized_1bit_qat.npz
    return 'student_fp32.pth', 'student_quantized_1bit_qat.npz'

def evaluate(npzfile):
    cmd = ['python3', 'quantized_qat_test.py']
    env = os.environ.copy(); env['QAT_NPZ']=npzfile
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # parse output for max diff and mean diff
    max_diff = None; mean_diff = None
    for line in out.stdout.splitlines():
        if 'Max diff:' in line:
            max_diff = float(line.split(':')[-1].strip())
        if 'Mean diff:' in line:
            mean_diff = float(line.split(':')[-1].strip())
    return max_diff, mean_diff

def main():
    os.makedirs('analysis', exist_ok=True)
    param_grid = [
        {'epochs':3,'lr':1e-3,'learnable':False},
        {'epochs':6,'lr':1e-3,'learnable':False},
        {'epochs':6,'lr':5e-4,'learnable':True},
        {'epochs':10,'lr':5e-4,'learnable':True},
    ]
    rows=[]
    for p in param_grid:
        print('Running QAT', p)
        _, npzfile = run_qat(p['epochs'], p['lr'], 16, p['learnable'])
        time.sleep(0.5)
        maxdiff, meandiff = evaluate(npzfile)
        # read saved metrics if exist
        metrics_csv = 'student_metrics.csv'
        max_train_loss = None
        if os.path.exists(metrics_csv):
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            max_train_loss = df['loss'].max()
        rows.append([p['epochs'], p['lr'], p['learnable'], maxdiff, meandiff, max_train_loss])
        rows.append([p['epochs'], p['lr'], p['learnable'], maxdiff, meandiff])
    with open('analysis/qat_grid.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epochs','lr','learnable','maxdiff','meandiff','max_train_loss'])
        w.writerows(rows)
    print('Wrote analysis/qat_grid.csv')

if __name__ == '__main__':
    main()
