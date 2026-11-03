# --------------------------------------------------------
# Task Identifier Variants for Ablation Study
# Supports: Quantum, Cosine, MLP, Attention, RBF
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from backbone.vit_quan_adapter import QuantumTaskIdentifier


class CosineTaskIdentifier(nn.Module):
    """基于余弦相似度的任务识别器"""
    def __init__(self, feature_dim: int = 768, max_tasks: int = 10, anchors_per_task: int = 3, ema_decay: float = 0.999, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_tasks = max_tasks
        self.anchors_per_task = anchors_per_task
        self.ema_decay = ema_decay

        # 任务锚点存储 [task_id][anchor_id] -> feature_vector
        self.register_buffer('task_anchors', torch.zeros(max_tasks, anchors_per_task, feature_dim))
        self.register_buffer('anchor_counts', torch.zeros(max_tasks, anchors_per_task))
        self.register_buffer('task_active', torch.zeros(max_tasks, dtype=torch.bool))
        self.register_buffer('epoch_sample_counts', torch.zeros(max_tasks, anchors_per_task))

        self.current_task_id = -1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def update_ema(self):
        """占位函数，保持接口一致"""
        pass

    def _init_ema_params(self):
        """占位函数，保持接口一致"""
        pass

    def reset_epoch_counts(self, task_id: int):
        self.epoch_sample_counts[task_id] = 0

    def forward(self, features, task_id, mode):
        """
        前向传播
        Args:
            features: 输入特征 [B, feature_dim]
            task_id: 当前任务ID
            mode: 'training' 或 'inference'
        """
        # L2归一化
        features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)

        if mode == 'training' and task_id is not None:
            self._update_task_anchors(features_norm, task_id)
            return features_norm

        elif mode == 'inference':
            predicted_tasks, confidences = self._compute_cosine_similarity(features_norm)
            return predicted_tasks, confidences

        return features_norm

    def _update_task_anchors(self, features: torch.Tensor, task_id: int):
        """更新任务锚点 - 基于余弦相似度的聚类"""
        with torch.no_grad():
            self.task_active[task_id] = True
            batch_size = features.size(0)

            for sample_idx in range(batch_size):
                sample_feature = features[sample_idx]

                if self.anchor_counts[task_id].sum() == 0:
                    # 初始化第一个锚点
                    self.task_anchors[task_id, 0] = sample_feature
                    self.anchor_counts[task_id, 0] = 1
                    self.epoch_sample_counts[task_id, 0] += 1
                else:
                    # 计算与现有锚点的余弦相似度
                    similarities = []
                    for anchor_idx in range(self.anchors_per_task):
                        if self.anchor_counts[task_id, anchor_idx] > 0:
                            anchor = self.task_anchors[task_id, anchor_idx]
                            similarity = torch.cosine_similarity(
                                sample_feature.unsqueeze(0),
                                anchor.unsqueeze(0),
                                dim=1
                            ).item()
                            similarities.append(similarity)
                        else:
                            similarities.append(-2)  # 未使用的锚点

                    max_similarity = max([s for s in similarities if s > -2], default=-1)

                    # 如果相似度较低，创建新锚点；否则更新现有锚点
                    if max_similarity < 0.7 and (self.anchor_counts[task_id] == 0).any():
                        empty_anchor_indices = (self.anchor_counts[task_id] == 0).nonzero(as_tuple=True)[0]
                        if len(empty_anchor_indices) > 0:
                            empty_anchor_idx = empty_anchor_indices[0].item()
                            self.task_anchors[task_id, empty_anchor_idx] = sample_feature
                            self.anchor_counts[task_id, empty_anchor_idx] = 1
                            self.epoch_sample_counts[task_id, empty_anchor_idx] += 1
                    else:
                        valid_similarities = [s for s in similarities if s > -2]
                        if len(valid_similarities) > 0:
                            best_anchor_idx = similarities.index(max(valid_similarities))
                            alpha = 0.1
                            updated_anchor = (
                                (1 - alpha) * self.task_anchors[task_id, best_anchor_idx] +
                                alpha * sample_feature
                            )
                            # 重新归一化
                            self.task_anchors[task_id, best_anchor_idx] = torch.nn.functional.normalize(
                                updated_anchor, p=2, dim=-1
                            )
                            self.epoch_sample_counts[task_id, best_anchor_idx] += 1

    def _compute_cosine_similarity(self, features: torch.Tensor):
        """计算与所有任务锚点的余弦相似度"""
        batch_size = features.size(0)
        predicted_tasks = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        confidences = torch.zeros(batch_size, device=features.device)

        active_tasks = torch.sum(self.task_active).item()
        if active_tasks <= 1:
            predicted_tasks.fill_(0)
            confidences.fill_(0.9)
            return predicted_tasks, confidences

        for sample_idx in range(batch_size):
            sample_feature = features[sample_idx:sample_idx+1]
            max_similarity = -1
            best_task = 0

            for task_id in range(self.max_tasks):
                if not self.task_active[task_id]:
                    continue

                task_max_similarity = -1
                for anchor_idx in range(self.anchors_per_task):
                    if self.anchor_counts[task_id, anchor_idx] > 0:
                        anchor = self.task_anchors[task_id, anchor_idx:anchor_idx+1]
                        similarity = torch.cosine_similarity(sample_feature, anchor, dim=1).item()
                        task_max_similarity = max(task_max_similarity, similarity)

                if task_max_similarity > max_similarity:
                    max_similarity = task_max_similarity
                    best_task = task_id

            predicted_tasks[sample_idx] = best_task
            confidences[sample_idx] = max(max_similarity, 0.1)

        return predicted_tasks, confidences

    def predict_task(self, features: torch.Tensor, threshold: float = 0.5):
        predicted_tasks, confidences = self.forward(features, task_id=None, mode='inference')
        if self.task_active.sum() <= 1:
            predicted_tasks.fill_(0)
        else:
            unknown_mask = confidences < threshold
            latest_task = torch.sum(self.task_active).item() - 1
            predicted_tasks[unknown_mask] = max(0, latest_task)
        return predicted_tasks, confidences

    def get_task_info(self):
        info = {}
        for task_id in range(self.max_tasks):
            if self.task_active[task_id]:
                anchor_counts = self.anchor_counts[task_id].cpu().numpy()
                epoch_counts = self.epoch_sample_counts[task_id].cpu().numpy()
                info[f'task_{task_id}'] = {
                    'active_anchors': (anchor_counts > 0).sum(),
                    'total_samples': int(epoch_counts.sum())
                }
        return info


class MLPTaskIdentifier(nn.Module):
    """基于MLP的任务识别器"""
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256, n_layers: int = 3,
                 max_tasks: int = 10, anchors_per_task: int = 3, ema_decay: float = 0.999, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_tasks = max_tasks
        self.anchors_per_task = anchors_per_task
        self.ema_decay = ema_decay

        # MLP编码器
        layers = []
        in_dim = feature_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # 任务锚点存储
        self.register_buffer('task_anchors', torch.zeros(max_tasks, anchors_per_task, hidden_dim))
        self.register_buffer('anchor_counts', torch.zeros(max_tasks, anchors_per_task))
        self.register_buffer('task_active', torch.zeros(max_tasks, dtype=torch.bool))
        self.register_buffer('epoch_sample_counts', torch.zeros(max_tasks, anchors_per_task))

        self.current_task_id = -1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def update_ema(self):
        """占位函数，保持接口一致"""
        pass

    def _init_ema_params(self):
        """占位函数，保持接口一致"""
        pass

    def reset_epoch_counts(self, task_id: int):
        self.epoch_sample_counts[task_id] = 0

    def forward(self, features, task_id, mode):
        """前向传播"""
        # 通过MLP编码
        encoded_features = self.encoder(features)
        # L2归一化
        encoded_features = torch.nn.functional.normalize(encoded_features, p=2, dim=-1)

        if mode == 'training' and task_id is not None:
            self._update_task_anchors(encoded_features, task_id)
            return encoded_features

        elif mode == 'inference':
            predicted_tasks, confidences = self._compute_similarity(encoded_features)
            return predicted_tasks, confidences

        return encoded_features

    def _update_task_anchors(self, features: torch.Tensor, task_id: int):
        """更新任务锚点"""
        with torch.no_grad():
            self.task_active[task_id] = True
            batch_size = features.size(0)

            for sample_idx in range(batch_size):
                sample_feature = features[sample_idx]

                if self.anchor_counts[task_id].sum() == 0:
                    self.task_anchors[task_id, 0] = sample_feature
                    self.anchor_counts[task_id, 0] = 1
                    self.epoch_sample_counts[task_id, 0] += 1
                else:
                    distances = []
                    for anchor_idx in range(self.anchors_per_task):
                        if self.anchor_counts[task_id, anchor_idx] > 0:
                            anchor = self.task_anchors[task_id, anchor_idx]
                            distance = torch.norm(sample_feature - anchor, p=2).item()
                            distances.append(distance)
                        else:
                            distances.append(float('inf'))

                    min_distance = min([d for d in distances if d != float('inf')], default=float('inf'))

                    if min_distance > 0.5 and (self.anchor_counts[task_id] == 0).any():
                        empty_anchor_indices = (self.anchor_counts[task_id] == 0).nonzero(as_tuple=True)[0]
                        if len(empty_anchor_indices) > 0:
                            empty_anchor_idx = empty_anchor_indices[0].item()
                            self.task_anchors[task_id, empty_anchor_idx] = sample_feature
                            self.anchor_counts[task_id, empty_anchor_idx] = 1
                            self.epoch_sample_counts[task_id, empty_anchor_idx] += 1
                    else:
                        valid_distances = [d for d in distances if d != float('inf')]
                        if len(valid_distances) > 0:
                            best_anchor_idx = distances.index(min(valid_distances))
                            alpha = 0.1
                            updated_anchor = (
                                (1 - alpha) * self.task_anchors[task_id, best_anchor_idx] +
                                alpha * sample_feature
                            )
                            self.task_anchors[task_id, best_anchor_idx] = torch.nn.functional.normalize(
                                updated_anchor, p=2, dim=-1
                            )
                            self.epoch_sample_counts[task_id, best_anchor_idx] += 1

    def _compute_similarity(self, features: torch.Tensor):
        """计算与所有任务锚点的相似度"""
        batch_size = features.size(0)
        predicted_tasks = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        confidences = torch.zeros(batch_size, device=features.device)

        active_tasks = torch.sum(self.task_active).item()
        if active_tasks <= 1:
            predicted_tasks.fill_(0)
            confidences.fill_(0.9)
            return predicted_tasks, confidences

        for sample_idx in range(batch_size):
            sample_feature = features[sample_idx:sample_idx+1]
            min_distance = float('inf')
            best_task = 0

            for task_id in range(self.max_tasks):
                if not self.task_active[task_id]:
                    continue

                task_min_distance = float('inf')
                for anchor_idx in range(self.anchors_per_task):
                    if self.anchor_counts[task_id, anchor_idx] > 0:
                        anchor = self.task_anchors[task_id, anchor_idx:anchor_idx+1]
                        distance = torch.norm(sample_feature - anchor, p=2).item()
                        task_min_distance = min(task_min_distance, distance)

                if task_min_distance < min_distance:
                    min_distance = task_min_distance
                    best_task = task_id

            predicted_tasks[sample_idx] = best_task
            # 距离转换为置信度
            confidences[sample_idx] = max(1.0 - min_distance, 0.1)

        return predicted_tasks, confidences

    def predict_task(self, features: torch.Tensor, threshold: float = 0.5):
        predicted_tasks, confidences = self.forward(features, task_id=None, mode='inference')
        if self.task_active.sum() <= 1:
            predicted_tasks.fill_(0)
        else:
            unknown_mask = confidences < threshold
            latest_task = torch.sum(self.task_active).item() - 1
            predicted_tasks[unknown_mask] = max(0, latest_task)
        return predicted_tasks, confidences

    def get_task_info(self):
        info = {}
        for task_id in range(self.max_tasks):
            if self.task_active[task_id]:
                anchor_counts = self.anchor_counts[task_id].cpu().numpy()
                epoch_counts = self.epoch_sample_counts[task_id].cpu().numpy()
                info[f'task_{task_id}'] = {
                    'active_anchors': (anchor_counts > 0).sum(),
                    'total_samples': int(epoch_counts.sum())
                }
        return info


class AttentionTaskIdentifier(nn.Module):
    """基于Attention的任务识别器"""
    def __init__(self, feature_dim: int = 768, num_heads: int = 8, n_layers: int = 3,
                 max_tasks: int = 10, anchors_per_task: int = 3, ema_decay: float = 0.999, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.max_tasks = max_tasks
        self.anchors_per_task = anchors_per_task
        self.ema_decay = ema_decay

        # Multi-head Self-Attention层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(n_layers)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.final_norm = nn.LayerNorm(feature_dim)

        # 任务锚点存储
        self.register_buffer('task_anchors', torch.zeros(max_tasks, anchors_per_task, feature_dim))
        self.register_buffer('anchor_counts', torch.zeros(max_tasks, anchors_per_task))
        self.register_buffer('task_active', torch.zeros(max_tasks, dtype=torch.bool))
        self.register_buffer('epoch_sample_counts', torch.zeros(max_tasks, anchors_per_task))

        self.current_task_id = -1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def update_ema(self):
        """占位函数，保持接口一致"""
        pass

    def _init_ema_params(self):
        """占位函数，保持接口一致"""
        pass

    def reset_epoch_counts(self, task_id: int):
        self.epoch_sample_counts[task_id] = 0

    def forward(self, features, task_id, mode):
        """前向传播"""
        # 添加batch维度用于attention
        x = features.unsqueeze(1)  # [B, 1, feature_dim]

        # 通过多层attention
        for i in range(self.n_layers):
            # Self-attention
            attn_out, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms[i](x + attn_out)

        # FFN
        x = self.final_norm(x + self.ffn(x))
        encoded_features = x.squeeze(1)  # [B, feature_dim]

        # L2归一化
        encoded_features = torch.nn.functional.normalize(encoded_features, p=2, dim=-1)

        if mode == 'training' and task_id is not None:
            self._update_task_anchors(encoded_features, task_id)
            return encoded_features

        elif mode == 'inference':
            predicted_tasks, confidences = self._compute_similarity(encoded_features)
            return predicted_tasks, confidences

        return encoded_features

    def _update_task_anchors(self, features: torch.Tensor, task_id: int):
        """更新任务锚点"""
        with torch.no_grad():
            self.task_active[task_id] = True
            batch_size = features.size(0)

            for sample_idx in range(batch_size):
                sample_feature = features[sample_idx]

                if self.anchor_counts[task_id].sum() == 0:
                    self.task_anchors[task_id, 0] = sample_feature
                    self.anchor_counts[task_id, 0] = 1
                    self.epoch_sample_counts[task_id, 0] += 1
                else:
                    similarities = []
                    for anchor_idx in range(self.anchors_per_task):
                        if self.anchor_counts[task_id, anchor_idx] > 0:
                            anchor = self.task_anchors[task_id, anchor_idx]
                            similarity = torch.cosine_similarity(
                                sample_feature.unsqueeze(0),
                                anchor.unsqueeze(0),
                                dim=1
                            ).item()
                            similarities.append(similarity)
                        else:
                            similarities.append(-2)

                    max_similarity = max([s for s in similarities if s > -2], default=-1)

                    if max_similarity < 0.7 and (self.anchor_counts[task_id] == 0).any():
                        empty_anchor_indices = (self.anchor_counts[task_id] == 0).nonzero(as_tuple=True)[0]
                        if len(empty_anchor_indices) > 0:
                            empty_anchor_idx = empty_anchor_indices[0].item()
                            self.task_anchors[task_id, empty_anchor_idx] = sample_feature
                            self.anchor_counts[task_id, empty_anchor_idx] = 1
                            self.epoch_sample_counts[task_id, empty_anchor_idx] += 1
                    else:
                        valid_similarities = [s for s in similarities if s > -2]
                        if len(valid_similarities) > 0:
                            best_anchor_idx = similarities.index(max(valid_similarities))
                            alpha = 0.1
                            updated_anchor = (
                                (1 - alpha) * self.task_anchors[task_id, best_anchor_idx] +
                                alpha * sample_feature
                            )
                            self.task_anchors[task_id, best_anchor_idx] = torch.nn.functional.normalize(
                                updated_anchor, p=2, dim=-1
                            )
                            self.epoch_sample_counts[task_id, best_anchor_idx] += 1

    def _compute_similarity(self, features: torch.Tensor):
        """计算与所有任务锚点的相似度"""
        batch_size = features.size(0)
        predicted_tasks = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        confidences = torch.zeros(batch_size, device=features.device)

        active_tasks = torch.sum(self.task_active).item()
        if active_tasks <= 1:
            predicted_tasks.fill_(0)
            confidences.fill_(0.9)
            return predicted_tasks, confidences

        for sample_idx in range(batch_size):
            sample_feature = features[sample_idx:sample_idx+1]
            max_similarity = -1
            best_task = 0

            for task_id in range(self.max_tasks):
                if not self.task_active[task_id]:
                    continue

                task_max_similarity = -1
                for anchor_idx in range(self.anchors_per_task):
                    if self.anchor_counts[task_id, anchor_idx] > 0:
                        anchor = self.task_anchors[task_id, anchor_idx:anchor_idx+1]
                        similarity = torch.cosine_similarity(sample_feature, anchor, dim=1).item()
                        task_max_similarity = max(task_max_similarity, similarity)

                if task_max_similarity > max_similarity:
                    max_similarity = task_max_similarity
                    best_task = task_id

            predicted_tasks[sample_idx] = best_task
            confidences[sample_idx] = max(max_similarity, 0.1)

        return predicted_tasks, confidences

    def predict_task(self, features: torch.Tensor, threshold: float = 0.5):
        predicted_tasks, confidences = self.forward(features, task_id=None, mode='inference')
        if self.task_active.sum() <= 1:
            predicted_tasks.fill_(0)
        else:
            unknown_mask = confidences < threshold
            latest_task = torch.sum(self.task_active).item() - 1
            predicted_tasks[unknown_mask] = max(0, latest_task)
        return predicted_tasks, confidences

    def get_task_info(self):
        info = {}
        for task_id in range(self.max_tasks):
            if self.task_active[task_id]:
                anchor_counts = self.anchor_counts[task_id].cpu().numpy()
                epoch_counts = self.epoch_sample_counts[task_id].cpu().numpy()
                info[f'task_{task_id}'] = {
                    'active_anchors': (anchor_counts > 0).sum(),
                    'total_samples': int(epoch_counts.sum())
                }
        return info


class RBFTaskIdentifier(nn.Module):
    """基于RBF (Radial Basis Function) 核的任务识别器"""
    def __init__(self, feature_dim: int = 768, max_tasks: int = 10, anchors_per_task: int = 3,
                 ema_decay: float = 0.999, gamma: float = 1.0, learnable_gamma: bool = True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_tasks = max_tasks
        self.anchors_per_task = anchors_per_task
        self.ema_decay = ema_decay
        self.learnable_gamma = learnable_gamma

        # RBF核参数 gamma (bandwidth)
        if learnable_gamma:
            # 可学习的gamma参数（每个任务一个）
            self.log_gamma = nn.Parameter(torch.zeros(max_tasks))
            print(f"Using learnable RBF gamma (per-task)")
        else:
            # 固定gamma
            self.register_buffer('gamma', torch.tensor(gamma))
            print(f"Using fixed RBF gamma = {gamma}")

        # 任务锚点存储
        self.register_buffer('task_anchors', torch.zeros(max_tasks, anchors_per_task, feature_dim))
        self.register_buffer('anchor_counts', torch.zeros(max_tasks, anchors_per_task))
        self.register_buffer('task_active', torch.zeros(max_tasks, dtype=torch.bool))
        self.register_buffer('epoch_sample_counts', torch.zeros(max_tasks, anchors_per_task))

        self.current_task_id = -1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def update_ema(self):
        """占位函数，保持接口一致"""
        pass

    def _init_ema_params(self):
        """占位函数，保持接口一致"""
        pass

    def reset_epoch_counts(self, task_id: int):
        self.epoch_sample_counts[task_id] = 0

    def forward(self, features, task_id, mode):
        """前向传播"""
        if mode == 'training' and task_id is not None:
            self._update_task_anchors(features, task_id)
            return features

        elif mode == 'inference':
            predicted_tasks, confidences = self._compute_rbf_similarity(features)
            return predicted_tasks, confidences

        return features

    def _update_task_anchors(self, features: torch.Tensor, task_id: int):
        """更新任务锚点 - 基于RBF核的聚类"""
        with torch.no_grad():
            self.task_active[task_id] = True
            batch_size = features.size(0)

            for sample_idx in range(batch_size):
                sample_feature = features[sample_idx]

                if self.anchor_counts[task_id].sum() == 0:
                    # 初始化第一个锚点
                    self.task_anchors[task_id, 0] = sample_feature
                    self.anchor_counts[task_id, 0] = 1
                    self.epoch_sample_counts[task_id, 0] += 1
                else:
                    # 计算与现有锚点的RBF相似度
                    rbf_similarities = []
                    for anchor_idx in range(self.anchors_per_task):
                        if self.anchor_counts[task_id, anchor_idx] > 0:
                            anchor = self.task_anchors[task_id, anchor_idx]
                            rbf_sim = self._compute_rbf_kernel(
                                sample_feature.unsqueeze(0),
                                anchor.unsqueeze(0),
                                task_id
                            ).item()
                            rbf_similarities.append(rbf_sim)
                        else:
                            rbf_similarities.append(-1)

                    max_similarity = max([s for s in rbf_similarities if s >= 0], default=0)

                    # RBF核值在[0,1]之间，阈值设为0.7
                    if max_similarity < 0.7 and (self.anchor_counts[task_id] == 0).any():
                        empty_anchor_indices = (self.anchor_counts[task_id] == 0).nonzero(as_tuple=True)[0]
                        if len(empty_anchor_indices) > 0:
                            empty_anchor_idx = empty_anchor_indices[0].item()
                            self.task_anchors[task_id, empty_anchor_idx] = sample_feature
                            self.anchor_counts[task_id, empty_anchor_idx] = 1
                            self.epoch_sample_counts[task_id, empty_anchor_idx] += 1
                    else:
                        valid_similarities = [s for s in rbf_similarities if s >= 0]
                        if len(valid_similarities) > 0:
                            best_anchor_idx = rbf_similarities.index(max(valid_similarities))
                            alpha = 0.1
                            updated_anchor = (
                                (1 - alpha) * self.task_anchors[task_id, best_anchor_idx] +
                                alpha * sample_feature
                            )
                            self.task_anchors[task_id, best_anchor_idx] = updated_anchor
                            self.epoch_sample_counts[task_id, best_anchor_idx] += 1

    def _compute_rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor, task_id: int = None):
        """
        计算RBF核（高斯核）
        K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        其中 gamma 控制核的宽度（bandwidth）
        """
        # 计算欧氏距离的平方
        squared_distance = torch.sum((x1 - x2) ** 2, dim=-1)

        # 获取gamma参数
        if self.learnable_gamma and task_id is not None:
            # 使用log_gamma确保gamma为正
            gamma = torch.exp(self.log_gamma[task_id])
        elif self.learnable_gamma:
            # 推理时使用所有任务的平均gamma
            gamma = torch.exp(self.log_gamma).mean()
        else:
            gamma = self.gamma

        # 计算RBF核
        rbf_value = torch.exp(-gamma * squared_distance)
        return rbf_value

    def _compute_rbf_similarity(self, features: torch.Tensor):
        """计算与所有任务锚点的RBF相似度"""
        batch_size = features.size(0)
        predicted_tasks = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        confidences = torch.zeros(batch_size, device=features.device)

        active_tasks = torch.sum(self.task_active).item()
        if active_tasks <= 1:
            predicted_tasks.fill_(0)
            confidences.fill_(0.9)
            return predicted_tasks, confidences

        for sample_idx in range(batch_size):
            sample_feature = features[sample_idx:sample_idx+1]
            max_similarity = -1
            best_task = 0

            for task_id in range(self.max_tasks):
                if not self.task_active[task_id]:
                    continue

                task_max_similarity = -1
                for anchor_idx in range(self.anchors_per_task):
                    if self.anchor_counts[task_id, anchor_idx] > 0:
                        anchor = self.task_anchors[task_id, anchor_idx:anchor_idx+1]
                        rbf_sim = self._compute_rbf_kernel(sample_feature, anchor, task_id).item()
                        task_max_similarity = max(task_max_similarity, rbf_sim)

                if task_max_similarity > max_similarity:
                    max_similarity = task_max_similarity
                    best_task = task_id

            predicted_tasks[sample_idx] = best_task
            confidences[sample_idx] = max(max_similarity, 0.1)

        return predicted_tasks, confidences

    def predict_task(self, features: torch.Tensor, threshold: float = 0.5):
        predicted_tasks, confidences = self.forward(features, task_id=None, mode='inference')
        if self.task_active.sum() <= 1:
            predicted_tasks.fill_(0)
        else:
            unknown_mask = confidences < threshold
            latest_task = torch.sum(self.task_active).item() - 1
            predicted_tasks[unknown_mask] = max(0, latest_task)
        return predicted_tasks, confidences

    def get_task_info(self):
        info = {}
        for task_id in range(self.max_tasks):
            if self.task_active[task_id]:
                anchor_counts = self.anchor_counts[task_id].cpu().numpy()
                epoch_counts = self.epoch_sample_counts[task_id].cpu().numpy()

                # 添加gamma信息
                if self.learnable_gamma:
                    gamma_value = torch.exp(self.log_gamma[task_id]).item()
                else:
                    gamma_value = self.gamma.item()

                info[f'task_{task_id}'] = {
                    'active_anchors': (anchor_counts > 0).sum(),
                    'total_samples': int(epoch_counts.sum()),
                    'gamma': gamma_value
                }
        return info
