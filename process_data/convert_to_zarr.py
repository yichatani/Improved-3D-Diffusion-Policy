import os
import re
import zarr
import numpy as np

data_path = '/home/ani/Downloads/routing'
save_path = '/home/ani/Downloads/routing.zarr'

zarr_root = zarr.open_group(save_path, mode='a')

# --- 创建 data 和 meta 分组 ---
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

# --- 检查示例 shape ---
first_episode = os.path.join(data_path, episode_folders[0])
rgb0 = np.load(os.path.join(first_episode, "rgb.npy"))
eef0 = np.load(os.path.join(first_episode, "eef_pose.npy"))

# --- 创建 dataset ---
zarr_data.create_dataset(
    'img',
    shape=(0, *rgb0.shape[1:]),
    chunks=(50, *rgb0.shape[1:]),
    dtype='uint8',
    compressor=compressor,
    overwrite=True
)
zarr_data.create_dataset(
    'state',
    shape=(0, eef0.shape[1]),
    chunks=(50, eef0.shape[1]),
    dtype='float32',
    compressor=compressor,
    overwrite=True
)
zarr_data.create_dataset(
    'action',
    shape=(0, eef0.shape[1]),
    chunks=(50, eef0.shape[1]),
    dtype='float32',
    compressor=compressor,
    overwrite=True
)

total_frames = 0
episode_end_indices = []
invalid_log = {}

for ep_idx, episode_folder in enumerate(episode_folders):
    episode_path = os.path.join(data_path, episode_folder)
    print(f"Processing {ep_idx+1}/{len(episode_folders)}: {episode_folder}")

    rgb = np.load(os.path.join(episode_path, "rgb.npy"))
    eef = np.load(os.path.join(episode_path, "eef_pose.npy"))

    # --- 确保长度一致 ---
    if rgb.shape[0] != eef.shape[0]:
        print(f"⚠️ Episode {episode_folder} frame mismatch: rgb={rgb.shape[0]}, eef={eef.shape[0]}")
        n = min(rgb.shape[0], eef.shape[0])
        rgb, eef = rgb[:n], eef[:n]

    # --- 检测异常帧 ---
    rgb_finite = np.all(np.isfinite(rgb.reshape(len(rgb), -1)), axis=1)
    eef_finite = np.all(np.isfinite(eef), axis=1)
    valid_mask = rgb_finite & eef_finite

    n_invalid = np.count_nonzero(~valid_mask)
    if n_invalid > 0:
        print(f"⚠️ {episode_folder}: Dropped {n_invalid}/{len(valid_mask)} invalid frames")
        invalid_log[episode_folder] = int(n_invalid)

    rgb = rgb[valid_mask]
    eef = eef[valid_mask]

    # --- 跳过长度太短的 episode ---
    if len(eef) < 2:
        print(f"⚠️ Skipping {episode_folder}: not enough valid frames ({len(eef)})")
        continue

    # --- 构建 state / action 对 ---
    state = eef[:-1]
    action = eef[1:]
    rgb = rgb[:-1]  # 对齐成与 state 一样多的帧

    n_frames = len(state)

    # --- 写入 Zarr ---
    zarr_data['img'].resize((total_frames + n_frames, *rgb.shape[1:]))
    zarr_data['state'].resize((total_frames + n_frames, state.shape[1]))
    zarr_data['action'].resize((total_frames + n_frames, action.shape[1]))

    zarr_data['img'][total_frames:total_frames + n_frames] = rgb
    zarr_data['state'][total_frames:total_frames + n_frames] = state
    zarr_data['action'][total_frames:total_frames + n_frames] = action

    total_frames += n_frames
    episode_end_indices.append(total_frames)

# --- 保存 meta 信息 ---
zarr_meta.create_dataset(
    'episode_ends',
    data=np.array(episode_end_indices),
    dtype='int64',
    overwrite=True,
    compressor=compressor
)

# --- 保存 invalid log ---
if len(invalid_log) > 0:
    import json
    log_path = os.path.join(os.path.dirname(save_path), "invalid_frames_log.json")
    with open(log_path, 'w') as f:
        json.dump(invalid_log, f, indent=2)
    print(f"⚠️ Saved invalid frame log to: {log_path}")

print("✅ Data conversion completed successfully.")
print(f"Total valid state-action pairs: {total_frames}")
