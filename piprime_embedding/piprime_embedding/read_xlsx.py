"""Read xlsx and report stats for last row."""
import sys
import pandas as pd
import numpy as np

xlsx = sys.argv[1] if len(sys.argv) > 1 else 'QLEEAEEEAQR_28.xlsx'
df = pd.read_excel(xlsx, sheet_name='cosine_similarity', index_col=0)
print(f'File: {xlsx}')
print(f'Shape: {df.shape}')

last_row = df.iloc[-1]
print(f'\nLast row (model prediction):')
print(f'  mean_excluding_self: {last_row["mean_excluding_self"]:.4f}')

sims = last_row.iloc[1:-1].values.astype(float)
sims_no_self = sims[sims < 0.999]  # exclude self-match
print(f'  vs other spectra:')
print(f'    min:  {sims_no_self.min():.4f}')
print(f'    max:  {sims_no_self.max():.4f}')
print(f'    mean: {sims_no_self.mean():.4f}')

all_means = df['mean_excluding_self'].iloc[:-1].values.astype(float)
print(f'\nReal spectra (first {len(all_means)} rows):')
print(f'  avg mean_excluding_self: {all_means.mean():.4f}')
