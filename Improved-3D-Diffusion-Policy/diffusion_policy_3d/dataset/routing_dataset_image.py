from typing import Dict
import torch
import numpy as np
import copy
import gc
from tqdm import tqdm
from termcolor import cprint

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class RoutingDatasetZarr(BaseDataset):
    """
    Dataset for robot routing task with explicit state and action in zarr.
    - state: absolute eef pose at each frame
    - action: absolute next-pose or control target (already stored)
    - Training target: relative action = action - obs_last
    """
    def __init__(self,
            zarr_path, 
            horizon=4,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_img=True,
            use_depth=False):
        super().__init__()
        cprint(f'Loading RoutingDatasetZarr from {zarr_path}', 'green')
        self.task_name = task_name
        self.use_img = use_img
        self.use_depth = use_depth
        self.n_obs_steps = pad_before + 1
        self.horizon = horizon

        # zarr keys
        buffer_keys = ['state', 'action']
        if self.use_img:
            buffer_keys.append('img')

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

        # dataset split
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
        self.pad_before = pad_before
        self.pad_after = pad_after

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
        """
        Compute normalizer using relative actions (action - last observed state)
        but use LinearNormalizer.fit for convenience.
        """
        rel_actions = []

        for ep_idx in tqdm(range(self.replay_buffer.n_episodes), desc="Computing relative actions"):
            if not self.train_mask[ep_idx]:
                continue

            start_idx = self.replay_buffer.episode_ends[ep_idx - 1] if ep_idx > 0 else 0
            end_idx = self.replay_buffer.episode_ends[ep_idx]

            states = self.replay_buffer['state'][start_idx:end_idx].astype(np.float32)
            actions = self.replay_buffer['action'][start_idx:end_idx].astype(np.float32)

            for i in range(len(actions) - self.horizon + 1):
                window_action = actions[i:i + self.horizon]
                window_state = states[i:i + self.horizon]
                last_obs = window_state[self.n_obs_steps - 1]
                rel_action = window_action - last_obs[None, :]
                rel_actions.append(rel_action)

        if len(rel_actions) == 0:
            cprint("No training data found, using identity normalizer.", "red")
            normalizer = LinearNormalizer()
            for k in ['action', 'agent_pos']:
                normalizer[k] = SingleFieldLinearNormalizer.create_identity()
            if self.use_img:
                normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
            if self.use_depth:
                normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
            return normalizer

        rel_actions = np.concatenate(rel_actions, axis=0)  # shape (N_total, action_dim)

        data = {'action': rel_actions}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        del rel_actions, data
        gc.collect()

        if self.use_img:
            normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
        # normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    # -----------------------------------------------------------
    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Convert one sampled sequence into model-ready data.
        - obs['agent_pos']: state
        - action: relative = stored action - last observed state
        """
        state = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        image = sample['img'].astype(np.float32)

        last_obs_state = state[self.n_obs_steps - 1].copy()
        rel_action = action - last_obs_state[None, :]

        data = {
            'obs': {'image': image},
            'action': rel_action
        }
        # if self.use_img:
        #     data['obs']['image'] = sample['img'].astype(np.float32)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data






# -----------------------------------------------------------
    # def get_normalizer(self, mode='limits', **kwargs):
    #     """Memory-safe normalizer (streaming version)"""
    #     n_total = 0
    #     sum_actions = None
    #     sum_sq_actions = None
    #     min_action = None
    #     max_action = None

    #     for ep_idx in tqdm(range(self.replay_buffer.n_episodes), desc="Computing normalizer stats"):
    #         if not self.train_mask[ep_idx]:
    #             continue

    #         start_idx = self.replay_buffer.episode_ends[ep_idx - 1] if ep_idx > 0 else 0
    #         end_idx = self.replay_buffer.episode_ends[ep_idx]
    #         state_ep = self.replay_buffer['state'][start_idx:end_idx].astype(np.float32)
    #         action_ep = self.replay_buffer['action'][start_idx:end_idx].astype(np.float32)

    #         for i in range(len(action_ep) - self.horizon + 1):
    #             window_action = action_ep[i:i + self.horizon]
    #             window_state = state_ep[i:i + self.horizon]
    #             last_obs = window_state[self.n_obs_steps - 1]
    #             rel_action = window_action - last_obs[None, :]

    #             # 更新统计量
    #             if min_action is None:
    #                 min_action = rel_action.min(axis=0)
    #                 max_action = rel_action.max(axis=0)
    #                 sum_actions = rel_action.sum(axis=0)
    #                 sum_sq_actions = (rel_action ** 2).sum(axis=0)
    #                 n_total = rel_action.shape[0]
    #             else:
    #                 min_action = np.minimum(min_action, rel_action.min(axis=0))
    #                 max_action = np.maximum(max_action, rel_action.max(axis=0))
    #                 sum_actions += rel_action.sum(axis=0)
    #                 sum_sq_actions += (rel_action ** 2).sum(axis=0)
    #                 n_total += rel_action.shape[0]

    #     if n_total == 0:
    #         cprint("⚠️ No training data, using identity normalizer.", "red")
    #         normalizer = LinearNormalizer()
    #         for k in ['action', 'agent_pos']:
    #             normalizer[k] = SingleFieldLinearNormalizer.create_identity()
    #         if self.use_img:
    #             normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
    #         return normalizer

    #     mean = sum_actions / n_total
    #     var = sum_sq_actions / n_total - mean ** 2
    #     std = np.sqrt(np.maximum(var, 1e-12))

    #     # 使用 numpy 保存参数
    #     normalizer = LinearNormalizer()
    #     action_norm = SingleFieldLinearNormalizer()

    #     if mode == 'limits':
    #         range_val = np.maximum(max_action - min_action, 1e-8)
    #         scale = 2.0 / range_val
    #         offset = -(max_action + min_action) / range_val
    #     elif mode == 'gaussian':
    #         scale = 1.0 / (std + 1e-8)
    #         offset = -mean / (std + 1e-8)
    #     else:
    #         raise ValueError(f"Unknown normalization mode: {mode}")

    #     action_norm._params_dict = dict(scale=scale, offset=offset)
    #     action_norm._input_stats_dict = dict(
    #         min=min_action, 
    #         max=max_action, 
    #         mean=mean, 
    #         std=std
    #     )

    #     normalizer['action'] = action_norm

    #     if self.use_img:
    #         normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
    #     # normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
    #     return normalizer















# from typing import Dict
# import torch
# import numpy as np
# import copy
# from tqdm import tqdm
# from termcolor import cprint

# from diffusion_policy_3d.common.pytorch_util import dict_apply
# from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
# from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
# from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
# from diffusion_policy_3d.dataset.base_dataset import BaseDataset


# class RoutingDatasetZarr(BaseDataset):
#     """
#     Dataset for robot routing task with explicit state and action in zarr.
#     - state: absolute eef pose at each frame
#     - action: absolute next-pose or control target (already stored)
#     - Training target: relative action = action - obs_last
#     """
#     def __init__(self,
#             zarr_path, 
#             horizon=4,
#             pad_before=0,
#             pad_after=0,
#             seed=42,
#             val_ratio=0.0,
#             max_train_episodes=None,
#             task_name=None,
#             use_img=True,
#             use_depth=False):
#         super().__init__()
#         cprint(f'Loading RoutingDatasetZarr from {zarr_path}', 'green')
#         self.task_name = task_name
#         self.use_img = use_img
#         self.n_obs_steps = pad_before + 1
#         self.horizon = horizon

#         # keys present in zarr
#         buffer_keys = ['state', 'action']
#         if self.use_img:
#             buffer_keys.append('img')

#         # 加载 replay buffer
#         self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

#         # mask 分割
#         val_mask = get_val_mask(
#             n_episodes=self.replay_buffer.n_episodes, 
#             val_ratio=val_ratio,
#             seed=seed)
#         train_mask = ~val_mask
#         train_mask = downsample_mask(
#             mask=train_mask, 
#             max_n=max_train_episodes, 
#             seed=seed)

#         # sampler
#         self.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer, 
#             sequence_length=horizon,
#             pad_before=pad_before, 
#             pad_after=pad_after,
#             episode_mask=train_mask)

#         self.train_mask = train_mask
#         self.pad_before = pad_before
#         self.pad_after = pad_after

#     # -----------------------------------------------------------
#     def get_validation_dataset(self):
#         val_set = copy.copy(self)
#         val_set.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer, 
#             sequence_length=self.horizon,
#             pad_before=self.pad_before, 
#             pad_after=self.pad_after,
#             episode_mask=~self.train_mask
#         )
#         val_set.train_mask = ~self.train_mask
#         return val_set

#     # -----------------------------------------------------------
#     def get_normalizer(self, mode='limits', **kwargs):
#         """Compute normalization stats from stored action data."""
#         n_total = 0
#         sum_actions = None
#         sum_sq_actions = None
#         min_action = None
#         max_action = None

#         for ep_idx in tqdm(range(self.replay_buffer.n_episodes), desc="Computing normalizer stats"):
#             if not self.train_mask[ep_idx]:
#                 continue

#             start_idx = self.replay_buffer.episode_ends[ep_idx - 1] if ep_idx > 0 else 0
#             end_idx = self.replay_buffer.episode_ends[ep_idx]
#             state_ep = self.replay_buffer['state'][start_idx:end_idx].astype(np.float32)
#             action_ep = self.replay_buffer['action'][start_idx:end_idx].astype(np.float32)

#             for i in range(len(action_ep) - self.horizon + 1):
#                 window_action = action_ep[i:i + self.horizon]
#                 window_state = state_ep[i:i + self.horizon]

#                 last_obs = window_state[self.n_obs_steps - 1]
#                 rel_action = window_action - last_obs[None, :]  # 相对动作

#                 if min_action is None:
#                     min_action = rel_action.min(axis=0)
#                     max_action = rel_action.max(axis=0)
#                     sum_actions = rel_action.sum(axis=0)
#                     sum_sq_actions = (rel_action ** 2).sum(axis=0)
#                     n_total = rel_action.shape[0]
#                 else:
#                     min_action = np.minimum(min_action, rel_action.min(axis=0))
#                     max_action = np.maximum(max_action, rel_action.max(axis=0))
#                     sum_actions += rel_action.sum(axis=0)
#                     sum_sq_actions += (rel_action ** 2).sum(axis=0)
#                     n_total += rel_action.shape[0]

#         if min_action is None or n_total == 0:
#             cprint("Warning: No training data for normalizer, using identity!", "red")
#             normalizer = LinearNormalizer()
#             for k in ['action', 'image', 'agent_pos']:
#                 normalizer[k] = SingleFieldLinearNormalizer.create_identity()
#             return normalizer

#         mean = sum_actions / n_total
#         var = sum_sq_actions / n_total - mean**2
#         std = np.sqrt(np.maximum(var, 1e-12))

#         normalizer = LinearNormalizer()
#         action_norm = SingleFieldLinearNormalizer()

#         if mode == 'limits':
#             range_val = max_action - min_action
#             range_val = np.maximum(range_val, 1e-8)
#             scale = 2.0 / range_val
#             offset = -(max_action + min_action) / range_val
#         elif mode == 'gaussian':
#             scale = 1.0 / (std + 1e-8)
#             offset = -mean / (std + 1e-8)
#         else:
#             raise ValueError(f"Unknown normalization mode: {mode}")

#         action_norm.params_dict = {'scale': scale, 'offset': offset}
#         action_norm.input_stats_dict = {
#             'min': min_action, 'max': max_action,
#             'mean': mean, 'std': std
#         }
#         normalizer['action'] = action_norm

#         if self.use_img:
#             normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
#         normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
#         return normalizer

#     # -----------------------------------------------------------
#     def __len__(self):
#         return len(self.sampler)

#     def _sample_to_data(self, sample):
#         """
#         Convert one sampled sequence into model-ready data.
#         - obs['agent_pos']: state
#         - action: relative = stored action - last observed state
#         """
#         state = sample['state'].astype(np.float32)
#         action = sample['action'].astype(np.float32)

#         # 相对动作（基于最后一帧观测）
#         last_obs_state = state[self.n_obs_steps - 1].copy()
#         rel_action = action - last_obs_state[None, :]

#         data = {
#             'obs': {'agent_pos': state},
#             'action': rel_action
#         }

#         if self.use_img:
#             data['obs']['image'] = sample['img'].astype(np.float32)

#         return data

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         sample = self.sampler.sample_sequence(idx)
#         data = self._sample_to_data(sample)
#         torch_data = dict_apply(data, torch.from_numpy)
#         return torch_data
