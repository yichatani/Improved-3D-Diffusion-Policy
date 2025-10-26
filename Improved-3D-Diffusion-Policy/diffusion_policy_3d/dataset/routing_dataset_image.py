from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
from termcolor import cprint
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RoutingDatasetZarr(BaseDataset):
    """
    Dataset for robot manipulation using eef_pose as state,
    and action defined as relative offset to the last observation frame (n_obs_steps-1).
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
        self.n_obs_steps = pad_before + 1
        self.horizon = horizon

        buffer_keys = ['eef_pose']
        if self.use_img:
            buffer_keys.append('img')

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

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
        n_total = 0
        sum_actions = None
        sum_sq_actions = None
        min_action = None
        max_action = None

        for ep_idx in tqdm(range(self.replay_buffer.n_episodes), desc="Computing normalizer stats"):
            if not self.train_mask[ep_idx]:
                continue

            start_idx = self.replay_buffer.episode_ends[ep_idx - 1] if ep_idx > 0 else 0
            end_idx = self.replay_buffer.episode_ends[ep_idx]
            eef_pose_ep = self.replay_buffer['eef_pose'][start_idx:end_idx].astype(np.float32)

            for i in range(len(eef_pose_ep) - self.horizon + 1):
                window = eef_pose_ep[i:i + self.horizon]
                if len(window) < self.horizon:
                    continue
                    
                last_obs = window[self.n_obs_steps - 1]
                action = window - last_obs[None, :]  # shape: (horizon, D)

                if min_action is None:
                    action_dim = action.shape[-1]
                    min_action = action.min(axis=0)
                    max_action = action.max(axis=0)
                    sum_actions = action.sum(axis=0)
                    sum_sq_actions = (action ** 2).sum(axis=0)
                    n_total = action.shape[0]
                else:
                    min_action = np.minimum(min_action, action.min(axis=0))
                    max_action = np.maximum(max_action, action.max(axis=0))
                    sum_actions += action.sum(axis=0)
                    sum_sq_actions += (action ** 2).sum(axis=0)
                    n_total += action.shape[0]

        if min_action is None or n_total == 0:
            cprint("Warning: No training data for normalizer, using identity!", "red")
            normalizer = LinearNormalizer()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
            if self.use_img:
                normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
            return normalizer

        mean = sum_actions / n_total
        var = sum_sq_actions / n_total - mean**2
        std = np.sqrt(np.maximum(var, 1e-12))

        normalizer = LinearNormalizer()
        action_normalizer = SingleFieldLinearNormalizer()
        
        if mode == 'limits':
            # normalized = (x - min) / (max - min) * 2 - 1
            #            = x * scale + offset
            # scale = 2 / (max - min)
            # offset = -1 - min * scale = -(max + min) / (max - min)
            range_val = max_action - min_action
            range_val = np.maximum(range_val, 1e-8)  # 避免除零
            
            scale = 2.0 / range_val
            offset = -(max_action + min_action) / range_val
            
            action_normalizer.params_dict = {
                'scale': scale,
                'offset': offset
            }
            action_normalizer.input_stats_dict = {
                'min': min_action,
                'max': max_action,
                'mean': mean,
                'std': std
            }
            
        elif mode == 'gaussian':
            # Z-score
            # normalized = (x - mean) / std
            #            = x * scale + offset
            # scale = 1 / std
            # offset = -mean / std
            scale = 1.0 / (std + 1e-8)
            offset = -mean / (std + 1e-8)
            
            action_normalizer.params_dict = {
                'scale': scale,
                'offset': offset
            }
            action_normalizer.input_stats_dict = {
                'min': min_action,
                'max': max_action,
                'mean': mean,
                'std': std
            }
            
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        
        normalizer['action'] = action_normalizer
        
        if self.use_img:
            normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        
        # cprint(f"Action normalizer stats (mode={mode}):", "cyan")
        # cprint(f"  Mean: {mean}", "cyan")
        # cprint(f"  Std:  {std}", "cyan")
        # cprint(f"  Min:  {min_action}", "cyan")
        # cprint(f"  Max:  {max_action}", "cyan")
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Convert one sampled sequence into model-ready data.
        - state: eef_pose
        - action: relative pose difference to obs_last (n_obs_steps-1)
        """
        eef_pose = sample['eef_pose'].astype(np.float32)

        last_obs_pose = eef_pose[self.n_obs_steps - 1].copy()

        action = eef_pose - last_obs_pose[None, :]

        data = {
            'obs': {'agent_pos': eef_pose},
            'action': action
        }

        if self.use_img:
            data['obs']['image'] = sample['img'].astype(np.float32)

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        torch_data = dict_apply(data, to_torch)
        return torch_data
