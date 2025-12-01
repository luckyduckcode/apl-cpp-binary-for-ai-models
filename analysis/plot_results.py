import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv('analysis/results.csv')
os.makedirs('docs/figures', exist_ok=True)

# Diff plot
plt.figure(figsize=(8,5))
for k, g in df.groupby('config'):
    plt.scatter([k], [g['max_diff'].mean()], label=k)
plt.title('Max diff by config')
plt.ylabel('Max diff')
plt.savefig('docs/figures/max_diff_by_config.png')
plt.close()

plt.figure(figsize=(8,5))
for k, g in df.groupby('config'):
    plt.scatter([k], [g['mean_diff'].mean()], label=k)
plt.title('Mean diff by config')
plt.ylabel('Mean diff')
plt.savefig('docs/figures/mean_diff_by_config.png')
plt.close()

# Latency plot
plt.figure(figsize=(8,5))
plt.bar(df['config'], df['fp32_ms'], label='FP32')
plt.bar(df['config'], df['dequant_ms'], alpha=0.6, label='Dequant')
plt.title('Avg latency across configs')
plt.ylabel('ms per batch')
plt.savefig('docs/figures/latency_by_config.png')
plt.close()

print('Saved plots to docs/figures')
