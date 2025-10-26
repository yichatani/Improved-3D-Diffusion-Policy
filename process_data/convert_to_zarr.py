import os
import re
import zarr
import numpy as np

data_path = '/home/ani/Downloads/routing'
save_path = '/home/ani/Downloads/routing.zarr'

zarr_root = zarr.open_group(save_path, mode='a')

if 'data' not in zarr_root:
    zarr_data = zarr_root.create_group('data')
    print("Created 'data' group")
else:
    zarr_data = zarr_root['data']
    print("'data' group already exists")

if 'meta' not in zarr_root:
    zarr_meta = zarr_root.create_group('meta')
    print("Created 'meta' group")
else:
    zarr_meta = zarr_root['meta']
    print("'meta' group already exists")

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
episode_pattern = re.compile(r'episode_(\d+)')
all_items = os.listdir(data_path)

episode_folders = sorted(
    [item for item in all_items if os.path.isdir(os.path.join(data_path, item)) and episode_pattern.match(item)],
    key=lambda x: int(episode_pattern.match(x).group(1))
)

print("Found episodes:", len(episode_folders))

total_frames = 0
episode_end_indices = []

first_episode = os.path.join(data_path, episode_folders[0])
rgb0 = np.load(os.path.join(first_episode, "rgb.npy"))
eef0 = np.load(os.path.join(first_episode, "eef_pose.npy"))

zarr_data.create_dataset(
    'img',
    shape=(0, *rgb0.shape[1:]),
    chunks=(50, *rgb0.shape[1:]),
    dtype='uint8',
    compressor=compressor,
    overwrite=True
)
zarr_data.create_dataset(
    'eef_pose',
    shape=(0, eef0.shape[1]),
    chunks=(50, eef0.shape[1]),
    dtype='float32',
    compressor=compressor,
    overwrite=True
)

for ep_idx, episode_folder in enumerate(episode_folders):
    episode_path = os.path.join(data_path, episode_folder)
    print(f"Processing {ep_idx+1}/{len(episode_folders)}: {episode_folder}")

    rgb = np.load(os.path.join(episode_path, "rgb.npy"))
    eef = np.load(os.path.join(episode_path, "eef_pose.npy"))

    n_frames = len(rgb)

    zarr_data['img'].resize((total_frames + n_frames, *rgb.shape[1:]))
    zarr_data['eef_pose'].resize((total_frames + n_frames, eef.shape[1]))

    zarr_data['img'][total_frames:total_frames + n_frames] = rgb
    zarr_data['eef_pose'][total_frames:total_frames + n_frames] = eef

    total_frames += n_frames
    episode_end_indices.append(total_frames)

zarr_meta.create_dataset(
    'episode_ends',
    data=np.array(episode_end_indices),
    dtype='int64',
    overwrite=True,
    compressor=compressor
)

print("✅ Data conversion completed successfully.")
print(f"Total frames: {total_frames}")
