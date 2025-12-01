import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('docs/figures', exist_ok=True)
df = pd.read_csv('analysis/qat_grid.csv')

plt.figure(figsize=(8,4))
df.plot(kind='line', x='epochs', y='maxdiff', marker='o')
plt.title('QAT Max Diff vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Max diff')
plt.savefig('docs/figures/qat_maxdiff.png')
plt.close()

plt.figure(figsize=(8,4))
df.plot(kind='line', x='epochs', y='meandiff', marker='o')
plt.title('QAT Mean Diff vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean diff')
plt.savefig('docs/figures/qat_meandiff.png')
plt.close()

plt.figure(figsize=(8,4))
df.plot(kind='line', x='epochs', y='max_train_loss', marker='o')
plt.title('QAT Max Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('docs/figures/qat_train_loss.png')
plt.close()

print('Wrote QAT grid plots to docs/figures')
