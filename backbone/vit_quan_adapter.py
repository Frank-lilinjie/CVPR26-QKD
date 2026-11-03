# --------------------------------------------------------
# Quantum Task Identification with Adapter for Class-Incremental Learning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed
import copy

# ============ Quantum Circuit Components ============

class QuantumGate(nn.Module):
    """基础量子门"""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

class ParameterizedQuantumGate(QuantumGate):
    """参数化量子旋转门"""
    def __init__(self, n_qubits: int, qubit_idx: int, gate_type: str = 'RY'):
        super().__init__(n_qubits)
        self.qubit_idx = qubit_idx
        self.gate_type = gate_type
        self.theta = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(0)
        device = state.device

        if self.gate_type == 'RY':
            cos_half = torch.cos(self.theta / 2)
            sin_half = torch.sin(self.theta / 2)
            gate_matrix = torch.tensor([
                [cos_half, -sin_half],
                [sin_half, cos_half]
            ], device=device, dtype=state.dtype)
        elif self.gate_type == 'RZ':
            exp_pos = torch.exp(1j * self.theta / 2)
            exp_neg = torch.exp(-1j * self.theta / 2)
            gate_matrix = torch.tensor([
                [exp_neg, 0],
                [0, exp_pos]
            ], device=device, dtype=state.dtype)
        else:
            raise ValueError(f"Unsupported gate type: {self.gate_type}")

        full_gate = self._construct_full_gate(gate_matrix, device, state.dtype)
        return torch.matmul(full_gate, state.unsqueeze(-1)).squeeze(-1)

    def _construct_full_gate(self, gate_matrix: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.n_qubits == 1:
            return gate_matrix
        ops = [torch.eye(2, device=device, dtype=dtype) for _ in range(self.n_qubits)]
        ops[self.qubit_idx] = gate_matrix
        result = ops[0]
        for op in ops[1:]:
            result = torch.kron(result, op)
        return result

class CNOTGate(QuantumGate):
    """CNOT门"""
    def __init__(self, n_qubits: int, control_qubit: int, target_qubit: int):
        super().__init__(n_qubits)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        device = state.device
        cnot_matrix = self._construct_cnot_matrix(device, state.dtype)
        return torch.matmul(cnot_matrix, state.unsqueeze(-1)).squeeze(-1)

    def _construct_cnot_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dim = 2 ** self.n_qubits
        cnot_matrix = torch.zeros((dim, dim), device=device, dtype=dtype)
        for i in range(dim):
            control_bit = (i >> (self.n_qubits - 1 - self.control_qubit)) & 1
            if control_bit == 1:
                target_flipped = i ^ (1 << (self.n_qubits - 1 - self.target_qubit))
                cnot_matrix[target_flipped, i] = 1
            else:
                cnot_matrix[i, i] = 1
        return cnot_matrix

class QuantumFeatureEncoder(nn.Module):
    """量子特征编码器"""
    def __init__(self, feature_dim: int, n_qubits: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self.feature_proj = nn.Linear(feature_dim, self.state_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_proj(features)
        angles = projected * math.pi
        amplitudes = torch.cos(angles) + 1j * torch.sin(angles)
        norm = torch.sqrt(torch.sum(torch.abs(amplitudes) ** 2, dim=-1, keepdim=True))
        quantum_state = amplitudes / (norm + 1e-8)
        return quantum_state

class QuantumTaskIdentifier(nn.Module):
    """量子任务识别器 - 基于锚点的自监督聚类，支持EMA和锚点分离损失，支持PQK"""
    def __init__(self, feature_dim: int = 768, n_qubits: int = 6, n_layers: int = 3, max_tasks: int = 10,
                 anchors_per_task: int = 3, ema_decay: float = 0.999, use_pqk: bool = False, n_projectors: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_tasks = max_tasks
        self.anchors_per_task = anchors_per_task
        self.state_dim = 2 ** n_qubits
        self.ema_decay = ema_decay
        self.use_pqk = use_pqk
        self.n_projectors = n_projectors

        self.encoder = QuantumFeatureEncoder(feature_dim, n_qubits)

        self.quantum_layers = nn.ModuleList()
        for layer in range(n_layers):
            layer_gates = nn.ModuleList()
            for qubit in range(n_qubits):
                layer_gates.append(ParameterizedQuantumGate(n_qubits, qubit, 'RY'))
                layer_gates.append(ParameterizedQuantumGate(n_qubits, qubit, 'RZ'))
            for qubit in range(n_qubits - 1):
                layer_gates.append(CNOTGate(n_qubits, qubit, qubit + 1))
            self.quantum_layers.append(layer_gates)

        self.ema_quantum_layers = nn.ModuleList()
        for layer in range(n_layers):
            ema_layer_gates = nn.ModuleList()
            for qubit in range(n_qubits):
                ema_layer_gates.append(ParameterizedQuantumGate(n_qubits, qubit, 'RY'))
                ema_layer_gates.append(ParameterizedQuantumGate(n_qubits, qubit, 'RZ'))
            for qubit in range(n_qubits - 1):
                ema_layer_gates.append(CNOTGate(n_qubits, qubit, qubit + 1))
            self.ema_quantum_layers.append(ema_layer_gates)

        # 初始化EMA参数
        self._init_ema_params()

        # EMA参数不参与梯度计算
        for ema_layer in self.ema_quantum_layers:
            for gate in ema_layer:
                for param in gate.parameters():
                    param.requires_grad = False

        # ========== PQK 相关组件 ==========
        if self.use_pqk:
            # 可学习的投影算子（测量基）
            # 每个投影算子是一个 state_dim x state_dim 的复数矩阵
            self.projectors = nn.ParameterList([
                nn.Parameter(torch.randn(self.state_dim, self.state_dim, dtype=torch.complex64) * 0.1)
                for _ in range(n_projectors)
            ])

            # 投影算子的权重
            self.projector_weights = nn.Parameter(torch.ones(n_projectors) / n_projectors)

            print(f"Using PQK with {n_projectors} projectors")
        else:
            print("Using standard fidelity-based quantum identifier")

        # 任务锚点存储 [task_id][anchor_id] -> quantum_state
        self.register_buffer('task_anchors', torch.zeros(max_tasks, anchors_per_task, self.state_dim, dtype=torch.complex64))
        self.register_buffer('anchor_counts', torch.zeros(max_tasks, anchors_per_task))
        self.register_buffer('task_active', torch.zeros(max_tasks, dtype=torch.bool))

        # 训练期间的样本统计（每个epoch重置）
        self.register_buffer('epoch_sample_counts', torch.zeros(max_tasks, anchors_per_task))

        # 当前活跃任务ID
        self.current_task_id = -1

    def _init_ema_params(self):
        """初始化EMA参数，复制当前参数到EMA模块"""
        with torch.no_grad():
            for layer_idx in range(self.n_layers):
                for gate_idx in range(len(self.quantum_layers[layer_idx])):
                    current_gate = self.quantum_layers[layer_idx][gate_idx]
                    ema_gate = self.ema_quantum_layers[layer_idx][gate_idx]

                    # 复制参数
                    if hasattr(current_gate, 'theta') and hasattr(ema_gate, 'theta'):
                        ema_gate.theta.data.copy_(current_gate.theta.data)

    def update_ema(self):
        """更新EMA参数，在每次训练步骤后调用"""
        with torch.no_grad():
            for layer_idx in range(self.n_layers):
                for gate_idx in range(len(self.quantum_layers[layer_idx])):
                    current_gate = self.quantum_layers[layer_idx][gate_idx]
                    ema_gate = self.ema_quantum_layers[layer_idx][gate_idx]

                    # EMA更新公式: ema_param = ema_decay * ema_param + (1 - ema_decay) * current_param
                    if hasattr(current_gate, 'theta') and hasattr(ema_gate, 'theta'):
                        ema_gate.theta.data.mul_(self.ema_decay).add_(
                            current_gate.theta.data, alpha=1 - self.ema_decay
                        )

    def set_task_id(self, task_id: int):
        """设置当前任务ID"""
        self.current_task_id = task_id

    def forward_with_ema(self, features, use_ema: bool = False):
        """使用EMA参数进行前向传播，用于评估旧任务性能"""
        quantum_state = self.encoder(features['features'])
        current_state = quantum_state

        # 选择使用当前参数还是EMA参数
        layers_to_use = self.ema_quantum_layers if use_ema else self.quantum_layers

        # 通过量子层
        for layer_gates in layers_to_use:
            for gate in layer_gates:
                current_state = gate(current_state)

        return current_state

    def forward(self, features, task_id, mode):
        """
        前向传播
        Args:
            features: 输入特征 [B, feature_dim]
            task_id: 当前任务ID（训练时使用）
            mode: 'training' 或 'inference'
        """
        quantum_state = self.encoder(features)
        current_state = quantum_state

        # 通过量子层
        for layer_gates in self.quantum_layers:
            for gate in layer_gates:
                current_state = gate(current_state)

        if mode == 'training' and task_id is not None:
            # 训练模式：更新当前任务的锚点
            self._update_task_anchors(current_state, task_id)
            return current_state

        elif mode == 'inference':
            # 推理模式：计算与所有锚点的保真度
            predicted_tasks, confidences = self._compute_anchor_fidelity(current_state)
            return predicted_tasks, confidences

        return current_state

    def _update_task_anchors(self, quantum_states: torch.Tensor, task_id: int):
        """更新任务锚点 - 自监督聚类"""
        with torch.no_grad():
            self.task_active[task_id] = True

            batch_size = quantum_states.size(0)

            for sample_idx in range(batch_size):
                sample_state = quantum_states[sample_idx]

                # 找到最接近的锚点进行更新
                if self.anchor_counts[task_id].sum() == 0:
                    # 初始化第一个锚点
                    self.task_anchors[task_id, 0] = sample_state
                    self.anchor_counts[task_id, 0] = 1
                    self.epoch_sample_counts[task_id, 0] += 1
                else:
                    # 计算与现有锚点的保真度
                    fidelities = []
                    for anchor_idx in range(self.anchors_per_task):
                        if self.anchor_counts[task_id, anchor_idx] > 0:
                            anchor = self.task_anchors[task_id, anchor_idx]
                            fidelity = self._compute_fidelity(sample_state.unsqueeze(0), anchor.unsqueeze(0))
                            fidelities.append(fidelity.item())
                        else:
                            fidelities.append(-1)  # 未使用的锚点

                    max_fidelity = max([f for f in fidelities if f >= 0], default=0)

                    # 如果保真度较低，创建新锚点；否则更新现有锚点
                    if max_fidelity < 0.7 and (self.anchor_counts[task_id] == 0).any():
                        # 创建新锚点
                        empty_anchor_indices = (self.anchor_counts[task_id] == 0).nonzero(as_tuple=True)[0]
                        if len(empty_anchor_indices) > 0:
                            empty_anchor_idx = empty_anchor_indices[0].item()
                            self.task_anchors[task_id, empty_anchor_idx] = sample_state
                            self.anchor_counts[task_id, empty_anchor_idx] = 1
                            self.epoch_sample_counts[task_id, empty_anchor_idx] += 1
                    else:
                        # 更新最接近的锚点 - 使用EMA更新
                        valid_fidelities = [f for f in fidelities if f >= 0]
                        if len(valid_fidelities) > 0:
                            best_anchor_idx = fidelities.index(max(valid_fidelities))
                            alpha = 0.1  # 学习率
                            updated_anchor = (
                                (1 - alpha) * self.task_anchors[task_id, best_anchor_idx] +
                                alpha * sample_state
                            )
                            norm = torch.norm(updated_anchor) + 1e-8  # 防止除零
                            self.task_anchors[task_id, best_anchor_idx] = updated_anchor / norm
                            # self.task_anchors[task_id, best_anchor_idx] = (
                            #     (1 - alpha) * self.task_anchors[task_id, best_anchor_idx] +
                            #     alpha * sample_state
                            # )
                            # 只在epoch级别统计样本数
                            self.epoch_sample_counts[task_id, best_anchor_idx] += 1

    def reset_epoch_counts(self, task_id: int):
        """重置指定任务的epoch样本计数"""
        self.epoch_sample_counts[task_id] = 0

    def reset_anchor_counts_for_task(self, task_id: int):
        """重置指定任务的锚点计数 - 在每个epoch开始时调用"""
        # 只重置计数，不重置锚点位置
        active_anchors = (self.anchor_counts[task_id] > 0).sum().item()
        if active_anchors > 0:
            # 保持活跃锚点的状态，但重置计数为1（表示该锚点存在）
            for anchor_idx in range(self.anchors_per_task):
                if self.anchor_counts[task_id, anchor_idx] > 0:
                    self.anchor_counts[task_id, anchor_idx] = 1

    def _compute_anchor_fidelity(self, quantum_states: torch.Tensor):
        """计算与所有任务锚点的保真度"""
        batch_size = quantum_states.size(0)
        predicted_tasks = torch.zeros(batch_size, dtype=torch.long, device=quantum_states.device)
        confidences = torch.zeros(batch_size, device=quantum_states.device)

        # 如果只有一个任务或没有任务，直接返回任务0和高置信度
        active_tasks = torch.sum(self.task_active).item()
        if active_tasks <= 1:
            predicted_tasks.fill_(0)
            confidences.fill_(0.9)  # 给一个较高的默认置信度
            return predicted_tasks, confidences

        for sample_idx in range(batch_size):
            sample_state = quantum_states[sample_idx:sample_idx+1]
            max_fidelity = -1
            best_task = 0

            # 遍历所有活跃任务的锚点
            for task_id in range(self.max_tasks):
                if not self.task_active[task_id]:
                    continue

                # 计算与该任务所有锚点的最大保真度
                task_max_fidelity = 0
                for anchor_idx in range(self.anchors_per_task):
                    if self.anchor_counts[task_id, anchor_idx] > 0:
                        anchor = self.task_anchors[task_id, anchor_idx:anchor_idx+1]
                        fidelity = self._compute_fidelity(sample_state, anchor)
                        task_max_fidelity = max(task_max_fidelity, fidelity.item())

                if task_max_fidelity > max_fidelity:
                    max_fidelity = task_max_fidelity
                    best_task = task_id

            predicted_tasks[sample_idx] = best_task
            confidences[sample_idx] = max(max_fidelity, 0.1)  # 确保最小置信度

        return predicted_tasks, confidences

    def _compute_fidelity(self, state1: torch.Tensor, state2: torch.Tensor):
        """计算两个量子态的保真度或PQK核"""
        if self.use_pqk:
            return self._compute_pqk_kernel(state1, state2)
        else:
            return self._compute_standard_fidelity(state1, state2)

    def _compute_standard_fidelity(self, state1: torch.Tensor, state2: torch.Tensor):
        """标准保真度计算：|<ψ1|ψ2>|²"""
        # 归一化
        state1 = state1 / (torch.norm(state1, dim=-1, keepdim=True) + 1e-8)
        state2 = state2 / (torch.norm(state2, dim=-1, keepdim=True) + 1e-8)

        # 计算保真度 |<ψ1|ψ2>|²
        overlap = torch.sum(torch.conj(state1) * state2, dim=-1)
        fidelity = torch.abs(overlap) ** 2
        return torch.mean(fidelity)

    def _compute_pqk_kernel(self, state1: torch.Tensor, state2: torch.Tensor):
        """PQK (Projected Quantum Kernel) 计算

        K(ψ1, ψ2) = Σ_i w_i * |<ψ1|P_i|ψ2>|²

        其中 P_i 是可学习的投影算子，w_i 是权重
        """
        # 归一化
        state1 = state1 / (torch.norm(state1, dim=-1, keepdim=True) + 1e-8)
        state2 = state2 / (torch.norm(state2, dim=-1, keepdim=True) + 1e-8)

        # Softmax归一化权重
        weights = torch.softmax(self.projector_weights, dim=0)

        kernel_value = 0.0
        for i, projector in enumerate(self.projectors):
            # 确保投影算子是厄米的（Hermitian）
            P = (projector + torch.conj(projector.T)) / 2

            # 计算 <ψ1|P|ψ2>
            # state1: [B, state_dim], P: [state_dim, state_dim], state2: [B, state_dim]
            # 先计算 P|ψ2>
            P_state2 = torch.matmul(state2, P.T)  # [B, state_dim]

            # 然后计算 <ψ1|P|ψ2>
            overlap = torch.sum(torch.conj(state1) * P_state2, dim=-1)  # [B]

            # 取模平方并加权
            projected_fidelity = torch.abs(overlap) ** 2
            kernel_value += weights[i] * projected_fidelity

        return torch.mean(kernel_value)

    def predict_task(self, features: torch.Tensor, threshold: float = 0.5):
        """预测任务ID"""
        predicted_tasks, confidences = self.forward(features, task_id=None, mode='inference')
        # 降低阈值，并且在只有一个任务时直接返回任务0
        if self.task_active.sum() <= 1:
            # 如果只有一个或没有活跃任务，直接预测为任务0
            predicted_tasks.fill_(0)
        else:
            # 低置信度的设为当前最新任务
            unknown_mask = confidences < threshold
            latest_task = torch.sum(self.task_active).item() - 1
            predicted_tasks[unknown_mask] = max(0, latest_task)
        return predicted_tasks, confidences

    def get_task_info(self):
        """获取任务信息统计"""
        info = {}
        for task_id in range(self.max_tasks):
            if self.task_active[task_id]:
                anchor_counts = self.anchor_counts[task_id].cpu().numpy()
                epoch_counts = self.epoch_sample_counts[task_id].cpu().numpy()
                info[f'task_{task_id}'] = {
                    'active_anchors': (anchor_counts > 0).sum(),
                    'total_samples': int(epoch_counts.sum())  # 当前epoch的样本数
                }
        return info

# ============ Adapter Components ============

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        # down = self.non_linear_func(down) # lora
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_option)

        x = residual + x
        return x

class VisionTransformer(nn.Module):
    """ViT with Quantum Task Identifier and Adapters"""
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 weight_init='', tuning_config=None):
        super().__init__()

        print("I'm using ViT with Quantum Task Identifier and Adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                config=tuning_config, layer_id=i
                )
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # ====== Task Identifier (支持多种类型) ======
        identifier_type = tuning_config.get('identifier_type', 'quantum')  # quantum/cosine/mlp/attention/rbf

        if identifier_type == 'quantum':
            self.quantum_identifier = QuantumTaskIdentifier(
                feature_dim=embed_dim,
                n_qubits=tuning_config.n_qubits,
                n_layers=tuning_config.quantum_layers,
                max_tasks=tuning_config.max_tasks,
                anchors_per_task=tuning_config.get('anchors_per_task', 3),
                ema_decay=tuning_config.get('ema_decay', 0.999),
                use_pqk=tuning_config.get('use_pqk', False),
                n_projectors=tuning_config.get('n_projectors', 4)
            )
        elif identifier_type == 'cosine':
            from backbone.task_identifiers import CosineTaskIdentifier
            self.quantum_identifier = CosineTaskIdentifier(
                feature_dim=embed_dim,
                max_tasks=tuning_config.max_tasks,
                anchors_per_task=tuning_config.get('anchors_per_task', 3),
                ema_decay=tuning_config.get('ema_decay', 0.999)
            )
        elif identifier_type == 'mlp':
            from backbone.task_identifiers import MLPTaskIdentifier
            self.quantum_identifier = MLPTaskIdentifier(
                feature_dim=embed_dim,
                hidden_dim=tuning_config.get('hidden_dim', 256),
                n_layers=tuning_config.get('identifier_layers', 3),
                max_tasks=tuning_config.max_tasks,
                anchors_per_task=tuning_config.get('anchors_per_task', 3),
                ema_decay=tuning_config.get('ema_decay', 0.999)
            )
        elif identifier_type == 'attention':
            from backbone.task_identifiers import AttentionTaskIdentifier
            self.quantum_identifier = AttentionTaskIdentifier(
                feature_dim=embed_dim,
                num_heads=tuning_config.get('num_heads', 8),
                n_layers=tuning_config.get('identifier_layers', 3),
                max_tasks=tuning_config.max_tasks,
                anchors_per_task=tuning_config.get('anchors_per_task', 3),
                ema_decay=tuning_config.get('ema_decay', 0.999)
            )
        elif identifier_type == 'rbf':
            from backbone.task_identifiers import RBFTaskIdentifier
            self.quantum_identifier = RBFTaskIdentifier(
                feature_dim=embed_dim,
                max_tasks=tuning_config.max_tasks,
                anchors_per_task=tuning_config.get('anchors_per_task', 3),
                ema_decay=tuning_config.get('ema_decay', 0.999),
                gamma=tuning_config.get('rbf_gamma', 1.0),
                learnable_gamma=tuning_config.get('learnable_gamma', True)
            )
        else:
            raise ValueError(f"Unknown identifier_type: {identifier_type}")

        print(f"Using {identifier_type.upper()} Task Identifier")

        # ====== Adapter Management ======
        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = nn.ModuleList()
        self.cur_adapter = nn.ModuleList()
        self.cur_task_id = 0  # 当前任务ID
        self.get_new_adapter()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def freeze(self):
        """冻结backbone，只保留adapter和量子电路可训练"""
        for param in self.parameters():
            param.requires_grad = False

        # adapter可训练
        for i in range(len(self.cur_adapter)):
            for param in self.cur_adapter[i].parameters():
                param.requires_grad = True

        # 量子电路可训练
        for param in self.quantum_identifier.parameters():
            param.requires_grad = True

    def get_new_adapter(self):
        config = self.config
        self.cur_adapter = nn.ModuleList()
        for i in range(len(self.blocks)):
            adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                init_option=config.ffn_adapter_init_option,
                                adapter_scalar=config.ffn_adapter_scalar,
                                adapter_layernorm_option=config.ffn_adapter_layernorm_option).to(self._device)
            self.cur_adapter.append(adapter)
        self.cur_adapter.requires_grad_(True)


    def add_adapter_to_list(self):
        """将当前adapter添加到列表并创建新adapter"""
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.cur_task_id += 1
        self.get_new_adapter()
    
    def forward_head(self, res):
        x = res['x']

        res['logits'] = self.head(x)
        return res
    
    def forward_train(self, x, quan=False):
        """训练阶段前向传播"""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过transformer blocks with current adapter
        if quan:
            x = self.blocks(x)
        else:
            for idx, blk in enumerate(self.blocks):
                x = blk(x, self.cur_adapter[idx])

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        res = dict()
        res['x'] = outcome
        res['pre_logits'] = outcome
        res['features'] = outcome
        return res

    def forward_train_quan(self, x):
        """训练阶段前向传播"""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):

            x = blk(x, self.adapter_list[0][idx])

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        res = dict()
        res['features'] = outcome
        return res

    def forward_test_with_quantum(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        # 先通过backbone获取特征用于量子识别
        x_temp = copy.deepcopy(x_init)
        for blk in self.blocks:
            x_temp = blk(x_temp, adapt=None)  # 不使用adapter
        x_temp = self.norm(x_temp)
        features_for_quantum = x_temp[:, 0]  # [B, embed_dim]
        # 量子任务识别：锚点保真度计算
        with torch.no_grad():
            predicted_tasks, confidence = self.quantum_identifier.predict_task(features_for_quantum)

        # 根据预测的任务使用对应的adapter
        outputs = []
        for b in range(B):
            task_id = predicted_tasks[b].item()

            # 处理未知任务：使用当前任务
            if task_id < 0 or task_id >= len(self.adapter_list) + 1:
                task_id = self.cur_task_id - 1 if self.cur_task_id > 0 else 0

            x_single = x_init[b:b+1]

            # 选择对应的adapter
            if task_id < len(self.adapter_list):
                adapters = self.adapter_list[task_id]
            else:
                adapters = self.cur_adapter

            # 通过blocks with selected adapter
            for i, blk in enumerate(self.blocks):
                x_single = blk(x_single, adapters[i])

            x_single = self.norm(x_single)
            outputs.append(x_single[:, 0])

        outcome = torch.cat(outputs, dim=0)

        res = dict()
        res['x'] = outcome
        res['predicted_tasks'] = predicted_tasks
        res['features'] = outcome
        return res

    def forward(self, x, test=False, fc_only=False, quan=False):
        if fc_only:
            res = dict()
            res['logits'] = self.head(x)
            return res
        if quan:
            res = self.forward_train_quan(x)
            return res
        if not test:
            res = self.forward_train(x)
            res = self.forward_head(res)
            return res
        else:
            res = self.forward_test_with_quantum(x)
            res = self.forward_head(res)
            return res

def vit_base_patch16_224_quan_adapter(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, checkpoint_path='/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

def vit_base_patch16_224_in21k_quan_adapter(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=False, num_classes=0, checkpoint_path='/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model