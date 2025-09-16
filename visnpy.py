import numpy as np

# Percorso del file .npy generato dal tuo modello
file_path = "/Users/gabriele/Desktop/Magistrale/Tesi/env/GT-GAN/generated_data_dumarey.npy"  # <-- sostituisci con il percorso corretto

# Carica i dati
generated_data = np.load(file_path)

# Informazioni generali
print(f"Shape del dataset: {generated_data.shape}")
print(f"Tipo di dati: {generated_data.dtype}")
print(f"Valori minimi: {np.min(generated_data)}")
print(f"Valori massimi: {np.max(generated_data)}")
print(f"Valore medio: {np.mean(generated_data)}")

# Stampa i primi 5 elementi della prima sequenza come esempio

print(generated_data)