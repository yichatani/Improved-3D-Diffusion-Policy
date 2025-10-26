#!/usr/bin/env python3
import numpy as np
import cv2
import os
import json
import zarr

zarr_path = "/home/ani/Downloads/routing.zarr"   # zarr file path
episode_id = 0                                 # show episode ID
# instruction_base = "/media/rmx/6C923B76923B43BC/data"  # original json instruction path

zarr_root = zarr.open_group(zarr_path, mode='r')
zarr_data = zarr_root['data']
zarr_meta = zarr_root['meta']

img_zarr = zarr_data['img']
eef_zarr = zarr_data['eef_pose']
episode_ends = np.array(zarr_meta['episode_ends'])

if episode_id == 0:
    start_idx = 0
else:
    start_idx = episode_ends[episode_id - 1]
end_idx = episode_ends[episode_id]

print(f"✅ Loaded episode_{episode_id}")
print(f" - Frame index range: [{start_idx}, {end_idx})")
print(f" - Total frames: {end_idx - start_idx}")

rgb_array = img_zarr[start_idx:end_idx]
eef_pose_array = eef_zarr[start_idx:end_idx]

print(f" - RGB shape: {rgb_array.shape}")
print(f" - EEF pose shape: {eef_pose_array.shape}")

# instruction_path = os.path.join(instruction_base, f"episode_{episode_id}", "instruction.json")
# if os.path.exists(instruction_path):
#     with open(instruction_path, "r") as f:
#         data = json.load(f)
#         print(f"🗣️ Instruction: {data.get('instruction', '')}")
# else:
#     print("⚠️ No instruction.json found for this episode.")

for i, (img, pose) in enumerate(zip(rgb_array, eef_pose_array)):
    x, y, z, qx, qy, qz, qw, width = pose

    img_display = img.copy()
    text = f"Frame {i}: Pos=({x:.3f}, {y:.3f}, {z:.3f}), Width={width*1000:.1f}mm"
    cv2.putText(img_display, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("RGB", img_display)
    key = cv2.waitKey(50)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
