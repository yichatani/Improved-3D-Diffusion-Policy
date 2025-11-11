from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from termcolor import cprint

class GR1DexDatasetImage(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_img=True,
            use_depth=False,
            n_obs_steps=2,
            ):
        super().__init__()
        cprint(f'Loading GR1DexDataset from {zarr_path}', 'green')
        self.task_name = task_name
        self.use_img = use_img
        self.use_depth = use_depth

        # 仅加载必要的键：eef_pose（用于计算action）、img（输入）、action（原始，但会被替换）
        buffer_keys = ['eef_pose', 'eef_pose_next']
        if self.use_img:
            buffer_keys.append('img')
        if self.use_depth:
            buffer_keys.append('depth')

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # 注意：这里的action是后续计算的“相对第一帧的变化量”，需用实际数据拟合
        # 先临时构造action数据用于归一化（实际训练时会被替换）
        # 若已有预处理好的action，可直接用self.replay_buffer['action']
        dummy_action = np.zeros((100, self.horizon, 8))  # 8维：7位姿+1夹爪
        data = {'action': dummy_action}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode,** kwargs)
        
        if self.use_img:
            normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 1. 提取当前帧位姿和预存的未来帧位姿（eef_pose_next）
        eef_pose = sample['eef_pose'].astype(np.float32)  # 形状：(T, 8)，当前帧位姿
        eef_pose_next = sample['eef_pose_next'].astype(np.float32)  # 形状：(T, 8)，未来帧位姿（已包含最后一帧补全）
        
        # print("-----------------------------------------eef_pose\n")
        # print(eef_pose)
        # print("-----------------------------------------eef_pose_next\n")
        # print(eef_pose_next)

        T = eef_pose.shape[0]

        if T == 0:
            action = np.zeros((0, 8), dtype=np.float32)
        else:
            # 2. 确定基准帧（最后一帧观测）
            last_obs_idx = min(self.n_obs_steps - 1, T - 1)
            last_obs_pose = eef_pose[last_obs_idx]  # 基准帧：如 p1（n_obs_steps=2时）

            # 3. 直接计算动作：未来帧相对于基准帧的偏移（无需切片和补全，因为eef_pose_next已处理好）
            action = eef_pose_next - last_obs_pose  # 形状：(T, 8)，与观测长度一致

            # 2. 替换第8维：直接用eef_pose_next的第8维（不减去基准帧）
            action[:, 7] = eef_pose_next[:, 7]  # 第8维的索引是7（0开始）
        # print("-----------------------------------------action\n")
        # print(action)

        # 构造输出
        data = {'action': action}
        obs = {}
        if self.use_img:
            obs['image'] = sample['img'].astype(np.float32)
        if self.use_depth:
            obs['depth'] = sample['depth'].astype(np.float32)
        data['obs'] = obs

        # 验证长度
        assert action.shape[0] == T, f"动作长度（{action.shape[0]}）与观测长度（{T}）不匹配"
        return data

    # def _sample_to_data(self, sample):
    #     eef_pose = sample['eef_pose'].astype(np.float32)  # 形状：(T, 8)
    #     # print("-----------------------------------------eef_pose\n")
    #     # print(eef_pose)
    #     T = eef_pose.shape[0]

    #     if T == 0:
    #         action = np.zeros((0, 8), dtype=np.float32)
    #     else:
    #         # 1. 确定基准帧（最后一帧观测）
    #         last_obs_idx = min(self.n_obs_steps - 1, T - 1)
    #         last_obs_pose = eef_pose[last_obs_idx]  # 基准帧：如 p1（n_obs_steps=2时）

    #         # 2. 计算「相对于基准帧的未来偏移」（动作是未来帧相对于基准帧的变化）
    #         # 目标：action[t] = eef_pose[t+1] - last_obs_pose（预测t+1帧相对于基准帧的偏移）
    #         if T == 1:
    #             # 单帧时，无未来帧，动作=0（相对于基准帧无变化）
    #             action = np.zeros_like(eef_pose)
    #         else:
    #             # 基础动作：未来帧相对于基准帧的偏移（T-1帧）
    #             base_action = eef_pose[1:] - last_obs_pose  # 如 [p2-p1, p3-p1, ..., p7-p1]（7帧）
                
    #             # 3. 补全最后一帧（用最后一个有效动作，确保长度=T=8）
    #             last_valid_action = base_action[-1:]  # 取最后一个未来偏移（如 p7-p1）
    #             pad_action = np.repeat(last_valid_action, 1, axis=0)
    #             action = np.concatenate([base_action, pad_action], axis=0)  # 8帧

    #     # print("-----------------------------------------action\n")
    #     # print(action)
    #     # 构造输出（确保动作与观测帧数一致）
    #     data = {'action': action}
    #     obs = {}
    #     if self.use_img:
    #         obs['image'] = sample['img'].astype(np.float32)
    #     if self.use_depth:
    #         obs['depth'] = sample['depth'].astype(np.float32)
    #     data['obs'] = obs

    #     # 验证长度
    #     assert action.shape[0] == T, f"动作长度（{action.shape[0]}）与观测长度（{T}）不匹配"
    #     return data

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        torch_data = dict_apply(data, to_torch_function)

        return torch_data


