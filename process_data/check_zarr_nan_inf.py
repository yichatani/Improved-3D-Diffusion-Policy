import zarr
import numpy as np
from tqdm import tqdm

save_path = '/root/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/data/routing.zarr'

z = zarr.open_group(save_path, mode='r')
data = z['data']

print("Checking Zarr Dataset NaN / Inf ...\n")

for key in ['img', 'state', 'action']:
    if key not in data:
        print(f"No '{key}' in the Dataset.")
        continue

    dset = data[key]
    print(f"Start to check '{key}' ... shape={dset.shape}")

    chunk_size = 5000
    n_total = dset.shape[0]
    n_nan, n_inf = 0, 0

    for start in tqdm(range(0, n_total, chunk_size), desc=f"Checking {key}", ncols=80):
        end = min(start + chunk_size, n_total)
        arr = dset[start:end]

        # transfer to float32 than check
        arr_flat = arr.reshape(-1).astype(np.float32)
        n_nan += np.count_nonzero(np.isnan(arr_flat))
        n_inf += np.count_nonzero(np.isinf(arr_flat))

    if n_nan > 0 or n_inf > 0:
        print(f"{key}: found NaN={n_nan}, Inf={n_inf}.")
    else:
        print(f"{key}: Not found NaN or Inf.")

print("\nFinish Checking!")
