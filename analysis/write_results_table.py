import pandas as pd
df = pd.read_csv('analysis/results.csv')
md = df.to_markdown(index=False)
with open('docs/results_table.md','w') as f:
    f.write('## Experiment Results Table\n\n')
    f.write(md)
print('Wrote docs/results_table.md')
