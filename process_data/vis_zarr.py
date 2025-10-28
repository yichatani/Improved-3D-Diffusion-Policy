#!/usr/bin/env python3
import numpy as np
import cv2
import os
import zarr

zarr_path = "/home/ani/Downloads/routing.zarr"   # zarr files path
episode_id = 0                                   # episode ID to be checked

zarr_root = zarr.open_group(zarr_path, mode='r')
zarr_data = zarr_root['data']
zarr_meta = zarr_root['meta']

if 'eef_pose' in zarr_data:
    img_zarr = zarr_data['img']
    eef_zarr = zarr_data['eef_pose']
elif 'state' in zarr_data:
    img_zarr = zarr_data['img']
    eef_zarr = zarr_data['state']
else:
    raise KeyError("No eef_pose or state data, please check Zarr structure.")

episode_ends = np.array(zarr_meta['episode_ends'])


if episode_id == 0:
    start_idx = 0
else:
    start_idx = episode_ends[episode_id - 1]
end_idx = episode_ends[episode_id]

print(f" - Loaded episode_{episode_id}")
print(f" - Frame index range: [{start_idx}, {end_idx})")
print(f" - Total frames: {end_idx - start_idx}")

rgb_array = img_zarr[start_idx:end_idx]
eef_pose_array = eef_zarr[start_idx:end_idx]

print(f" - RGB array shape: {rgb_array.shape}")   # (T, H, W, C)
print(f" - Single RGB frame shape: {rgb_array[0].shape}")  # (H, W, C)
print(f" - EEF pose shape: {eef_pose_array.shape}")

for i, (img, pose) in enumerate(zip(rgb_array, eef_pose_array)):
    if len(pose) == 8:
        x, y, z, qx, qy, qz, qw, width = pose
    else:
        x, y, z = pose[:3]
        width = np.nan

    img_display = img.copy()

    rgb_shape_text = f"RGB: {img.shape}"
    pos_text = f"Frame {i}: Position=({x:.3f}, {y:.3f}, {z:.3f})"
    if not np.isnan(width):
        pos_text += f", Width={width*1000:.1f}mm"

    cv2.putText(img_display, rgb_shape_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img_display, pos_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Episode Viewer", img_display)
    key = cv2.waitKey(50)
    if key == ord('q'):
        print("👋 Exiting viewer...")
        break

cv2.destroyAllWindows()
