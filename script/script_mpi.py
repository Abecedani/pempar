from mpi4py import MPI
import pandas as pd
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dataset_size = "1143"
dataset_path = f'../data/dataset_{dataset_size}.csv'

def clean_text(text):
    result = []
    for ch in text.lower():
        if ch.isalpha() or ch == ' ':
            result.append(ch)
    return ''.join(result)

if rank == 0:
    start_time = time.time()
    df = pd.read_csv(dataset_path, on_bad_lines='skip')
    df['teks'] = df['teks'].astype(str).apply(clean_text)
    chunks = np.array_split(df, size)
else:
    chunks = None

local_data = comm.scatter(chunks, root=0)

if isinstance(local_data, np.ndarray):
    local_df = pd.DataFrame(local_data, columns=['teks', 'label'])
else:
    local_df = local_data

local_word_counts = {0: {}, 1: {}, 2: {}}

for _, row in local_df.iterrows():
    label = int(row['label'])
    words = str(row['teks']).split()
    for word in words:
        if word:
            local_word_counts[label][word] = local_word_counts[label].get(word, 0) + 1

all_counts = comm.gather(local_word_counts, root=0)

if rank == 0:
    global_counts = {0: {}, 1: {}, 2: {}}
    for partial_count in all_counts:
        for label, words in partial_count.items():
            for word, count in words.items():
                global_counts[label][word] = global_counts[label].get(word, 0) + count

    end_time = time.time()

    print(f"--- Eksekusi MPI Selesai ---")
    print(f"Jumlah Data   : {dataset_size}")
    print(f"Jumlah Core   : {size}")
    print(f"Total waktu   : {end_time - start_time:.4f} detik")
    print(f"Status        : Model Frekuensi Kata Berhasil Dibangun secara Paralel")
