#!/usr/bin/env python3
import numpy as np
import cv2
import zarr
import math

# ------------------------------
# 基本设置
# ------------------------------
zarr_path = "/home/ani/Downloads/routing.zarr"   # zarr 文件路径
episode_ids = [0, 1, 2, 3, 4, 5]   # 可改成 [0,1,2,3,4,5,6,7] 显示 8 个
wait_ms = 25                # 每帧间隔（毫秒）

# ------------------------------
# 读取 zarr 数据
# ------------------------------
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

# ------------------------------
# 预读取所有 episode 数据
# ------------------------------
episodes = []
for episode_id in episode_ids:
    start_idx = 0 if episode_id == 0 else episode_ends[episode_id - 1]
    end_idx = episode_ends[episode_id]

    rgb_array = img_zarr[start_idx:end_idx]
    eef_pose_array = eef_zarr[start_idx:end_idx]
    episodes.append({
        'id': episode_id,
        'rgb': rgb_array,
        'pose': eef_pose_array,
        'length': len(rgb_array)
    })
    print(f"✅ Loaded episode_{episode_id} ({len(rgb_array)} frames)")

max_len = max(ep['length'] for ep in episodes)
print(f"📹 Max frame length among all episodes: {max_len}")

# ------------------------------
# 动态计算显示网格大小
# ------------------------------
n = len(episodes)
cols = math.ceil(math.sqrt(n))
rows = math.ceil(n / cols)
print(f"🧩 Display grid: {rows} x {cols}")

# ------------------------------
# 播放所有 episode
# ------------------------------
for frame_idx in range(max_len):
    frames = []
    for ep in episodes:
        # 获取当前帧
        if frame_idx < ep['length']:
            img = ep['rgb'][frame_idx].copy()
            pose = ep['pose'][frame_idx]
            if len(pose) >= 3:
                x, y, z = pose[:3]
            else:
                x = y = z = 0
            text = f"Ep{ep['id']} F{frame_idx} ({x:.2f},{y:.2f},{z:.2f})"
            cv2.putText(img, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 如果该 episode 已结束，用黑图占位
            h, w, c = episodes[0]['rgb'][0].shape
            img = np.zeros((h, w, c), dtype=np.uint8)
            cv2.putText(img, f"Ep{ep['id']} End", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        frames.append(img)

    # 将所有帧拼成网格
    h, w, c = frames[0].shape
    grid = []
    for r in range(rows):
        row_imgs = []
        for c_ in range(cols):
            idx = r * cols + c_
            if idx < len(frames):
                row_imgs.append(frames[idx])
            else:
                row_imgs.append(np.zeros_like(frames[0]))
        grid.append(np.hstack(row_imgs))
    grid_img = np.vstack(grid)

    cv2.imshow("Multi-Episode Viewer", grid_img)
    key = cv2.waitKey(wait_ms)
    if key == ord('q'):
        print("👋 Exiting viewer...")
        break

cv2.destroyAllWindows()





# #!/usr/bin/env python3
# import numpy as np
# import cv2
# import os
# import zarr

# # ------------------------------
# # 基本设置
# # ------------------------------
# zarr_path = "/home/ani/Downloads/routing.zarr"   # zarr 文件路径
# episode_id = 50                                   # 要查看的 episode ID

# # ------------------------------
# # 读取 zarr 数据
# # ------------------------------
# zarr_root = zarr.open_group(zarr_path, mode='r')
# zarr_data = zarr_root['data']
# zarr_meta = zarr_root['meta']

# # 支持旧版命名（eef_pose）或新版命名（state/action）
# if 'eef_pose' in zarr_data:
#     img_zarr = zarr_data['img']
#     eef_zarr = zarr_data['eef_pose']
# elif 'state' in zarr_data:
#     img_zarr = zarr_data['img']
#     eef_zarr = zarr_data['state']
# else:
#     raise KeyError("❌ 未找到 eef_pose 或 state 数据，请检查 Zarr 结构。")

# episode_ends = np.array(zarr_meta['episode_ends'])

# # ------------------------------
# # 计算当前 episode 范围
# # ------------------------------
# if episode_id == 0:
#     start_idx = 0
# else:
#     start_idx = episode_ends[episode_id - 1]
# end_idx = episode_ends[episode_id]

# print(f"✅ Loaded episode_{episode_id}")
# print(f" - Frame index range: [{start_idx}, {end_idx})")
# print(f" - Total frames: {end_idx - start_idx}")

# # ------------------------------
# # 读取数据
# # ------------------------------
# rgb_array = img_zarr[start_idx:end_idx]
# eef_pose_array = eef_zarr[start_idx:end_idx]

# print(f" - RGB array shape: {rgb_array.shape}")   # (T, H, W, C)
# print(f" - Single RGB frame shape: {rgb_array[0].shape}")  # (H, W, C)
# print(f" - EEF pose shape: {eef_pose_array.shape}")

# # ------------------------------
# # 播放
# # ------------------------------
# for i, (img, pose) in enumerate(zip(rgb_array, eef_pose_array)):
#     # 解析 eef_pose
#     if len(pose) == 8:
#         x, y, z, qx, qy, qz, qw, width = pose
#     else:
#         # 如果没有 width，就只打印位置
#         x, y, z = pose[:3]
#         width = np.nan

#     img_display = img.copy()

#     # 构建文字信息
#     rgb_shape_text = f"RGB: {img.shape}"
#     pos_text = f"Frame {i}: Position=({x:.3f}, {y:.3f}, {z:.3f})"
#     if not np.isnan(width):
#         pos_text += f", Width={width*1000:.1f}mm"

#     # 绘制文字
#     cv2.putText(img_display, rgb_shape_text, (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     cv2.putText(img_display, pos_text, (10, 55),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("Episode Viewer", img_display)
#     key = cv2.waitKey(50)
#     if key == ord('q'):
#         print("👋 Exiting viewer...")
#         break

# cv2.destroyAllWindows()


# # #!/usr/bin/env python3
# # import numpy as np
# # import cv2
# # import os
# # import json
# # import zarr

# # zarr_path = "/home/ani/Downloads/routing.zarr"   # zarr file path
# # episode_id = 0                                 # show episode ID
# # # instruction_base = "/media/rmx/6C923B76923B43BC/data"  # original json instruction path

# # zarr_root = zarr.open_group(zarr_path, mode='r')
# # zarr_data = zarr_root['data']
# # zarr_meta = zarr_root['meta']

# # img_zarr = zarr_data['img']
# # eef_zarr = zarr_data['eef_pose']
# # episode_ends = np.array(zarr_meta['episode_ends'])

# # if episode_id == 0:
# #     start_idx = 0
# # else:
# #     start_idx = episode_ends[episode_id - 1]
# # end_idx = episode_ends[episode_id]

# # print(f"✅ Loaded episode_{episode_id}")
# # print(f" - Frame index range: [{start_idx}, {end_idx})")
# # print(f" - Total frames: {end_idx - start_idx}")

# # rgb_array = img_zarr[start_idx:end_idx]
# # eef_pose_array = eef_zarr[start_idx:end_idx]

# # print(f" - RGB shape: {rgb_array.shape}")
# # print(f" - EEF pose shape: {eef_pose_array.shape}")

# # # instruction_path = os.path.join(instruction_base, f"episode_{episode_id}", "instruction.json")
# # # if os.path.exists(instruction_path):
# # #     with open(instruction_path, "r") as f:
# # #         data = json.load(f)
# # #         print(f"🗣️ Instruction: {data.get('instruction', '')}")
# # # else:
# # #     print("⚠️ No instruction.json found for this episode.")

# # for i, (img, pose) in enumerate(zip(rgb_array, eef_pose_array)):
# #     x, y, z, qx, qy, qz, qw, width = pose

# #     img_display = img.copy()
# #     text = f"Frame {i}: Pos=({x:.3f}, {y:.3f}, {z:.3f}), Width={width*1000:.1f}mm"
# #     cv2.putText(img_display, text, (10, 30),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# #     cv2.imshow("RGB", img_display)
# #     key = cv2.waitKey(50)
# #     if key == ord('q'):
# #         break

# # cv2.destroyAllWindows()
