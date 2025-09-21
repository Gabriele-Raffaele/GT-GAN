import numpy as np
import pandas as pd
import pathlib

# Percorso del file .npy generato
npy_file = "./dumarey_model/dumarey_pretrained/generated_data_dumarey.npy"
originale_dataset = "./datasets/dumarey.csv"
csv_file = pathlib.Path(npy_file).with_suffix(".csv")

# Carica i dati
data = np.load(npy_file)
print(f"Shape del dataset: {data.shape}")  # (num_sequences, seq_len, num_features)

num_sequences, seq_len, num_features = data.shape

# Carica il dataset originale per ottenere i nomi delle colonne
df_original = pd.read_csv(originale_dataset)
column_names = list(df_original.columns)
# Rimuovi colonne extra come 'sequence_id' e 'time_step' se presenti
column_names = [col for col in column_names if col not in ['sequence_id', 'time_step']]
feature_columns = [col for col in column_names if col not in ['sequence_id', 'time_step']]

# Creazione di una lista di dizionari per costruire il DataFrame
rows = []
for seq_id in range(num_sequences):
    for t in range(seq_len):
        row = {"sequence_id": seq_id, "time_step": t}
        for f in range(num_features):
            row[feature_columns[f]] = data[seq_id, t, f]
        rows.append(row)

# Creazione DataFrame e salvataggio in CSV
df = pd.DataFrame(rows)
df = df[["sequence_id", "time_step"] + feature_columns]
df.to_csv(csv_file, index=False)
print(f"CSV salvato in {csv_file}")