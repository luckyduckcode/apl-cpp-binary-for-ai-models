import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs('docs/figures', exist_ok=True)
df=pd.read_csv('analysis/xnor_bench.csv')
for t in sorted(df['threads'].unique()):
    sub=df[df['threads']==t]
    plt.figure()
    labels=sub['out'].astype(str)+"x"+sub['in'].astype(str)
    plt.plot(labels, sub['floatact_ms'], label='floatact', marker='o')
    plt.plot(labels, sub['binact_ms'], label='binact', marker='x')
    plt.xlabel('Size')
    plt.ylabel('Avg ms')
    plt.title(f'XNOR bench threads={t}')
    plt.legend()
    plt.savefig(f'docs/figures/xnor_threads_{t}.png')
    plt.close()

print('Wrote XNOR bench figures')
