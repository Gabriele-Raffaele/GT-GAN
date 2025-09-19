import numpy as np
import pandas as pd
import pathlib

# Percorso del file .npy generato
npy_file = "/Users/gabriele/Desktop/Magistrale/Tesi/env/GT-GAN/generated_data_dumarey.npy"
csv_file = pathlib.Path(npy_file).with_suffix(".csv")

# Carica i dati
data = np.load(npy_file)
print(f"Shape del dataset: {data.shape}")  # (num_sequences, seq_len, num_features)

num_sequences, seq_len, num_features = data.shape

# Creazione di una lista di dizionari per costruire il DataFrame
rows = []
for seq_id in range(num_sequences):
    for t in range(seq_len):
        row = {"sequence_id": seq_id, "time_step": t}
        for f in range(num_features):
            row[f"feature_{f+1}"] = data[seq_id, t, f]
        rows.append(row)

# Creazione DataFrame e salvataggio in CSV
df = pd.DataFrame(rows)
df.to_csv(csv_file, index=False)
print(f"CSV salvato in {csv_file}")