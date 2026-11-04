# --------------------------------------------------------
# Quantum Task Identification with Adapter for Class-Incremental Learning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed
import copy
import os
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    StandardScaler = None
    TruncatedSVD = None
    SKLEARN_AVAILABLE = False
import numpy as np
import logging
import pickle  # 添加pickle用于保存原型
from utils.device import resolve_quantum_device
from utils.paths import resolve_checkpoint_path

# ============ Feature Preprocessing Module ============

class FeaturePreprocessor(nn.Module):
    """Truncated SVD + 标准化预处理模块（集成到 PyTorch 模型中，无需外部文件）"""
    def __init__(self, input_dim=768, output_dim=12, random_state=1993):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_state = random_state
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.svd = TruncatedSVD(n_components=output_dim, random_state=random_state) if SKLEARN_AVAILABLE else None
        self.logger = logging.getLogger(__name__)

        # 使用 PyTorch buffer 保存预处理参数（随模型一起保存）
        self.register_buffer('fitted_flag', torch.tensor(0, dtype=torch.bool))

        # 预留 buffer 用于保存 scaler 参数
        self.register_buffer('scaler_mean', torch.zeros(input_dim))
        self.register_buffer('scaler_std', torch.ones(input_dim))

        # 预留 buffer 用于保存 SVD 参数
        self.register_buffer('svd_components', torch.zeros(output_dim, input_dim))
        self.register_buffer('svd_mean', torch.zeros(input_dim))

    def fit(self, features):
        """拟合预处理器 - 只在第一次调用时执行，后续调用直接返回 transform 结果"""
        # 如果已经 fitted，则直接使用 transform
        if self.fitted_flag.item():
            print(f"[INFO] Preprocessor already fitted, using existing parameters")
            return self.transform(features)

        X = features.detach().cpu().float() if torch.is_tensor(features) else torch.tensor(features, dtype=torch.float32)

        if SKLEARN_AVAILABLE:
            X_np = X.numpy()

            # 标准化
            X_scaled = self.scaler.fit_transform(X_np)

            # 保存 scaler 参数到 buffer
            self._copy_buffer(self.scaler_mean, torch.from_numpy(self.scaler.mean_).float())
            self._copy_buffer(self.scaler_std, torch.from_numpy(self.scaler.scale_).float())

            # L2 归一化
            X_normalized = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)

            # Truncated SVD 降维
            n_components = min(X_normalized.shape[-1], self.output_dim)
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            Z = self.svd.fit_transform(X_normalized)

            # 保存 SVD 参数到 buffer
            svd_components = torch.from_numpy(self.svd.components_).float()
            if svd_components.shape[0] < self.output_dim:
                padded = torch.zeros(self.output_dim, self.input_dim)
                padded[:svd_components.shape[0]] = svd_components
                self._copy_buffer(self.svd_components, padded)
            else:
                self._copy_buffer(self.svd_components, svd_components)

            # SVD 的 mean 可能不存在（TruncatedSVD 不中心化）
            if hasattr(self.svd, 'mean_') and self.svd.mean_ is not None:
                self._copy_buffer(self.svd_mean, torch.from_numpy(self.svd.mean_).float())

            # L2 归一化 SVD 输出
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        else:
            self.logger.warning("scikit-learn not found, using torch PCA fallback for feature preprocessing")

            mean = X.mean(dim=0)
            std = X.std(dim=0, unbiased=False).clamp_min(1e-8)
            self._copy_buffer(self.scaler_mean, mean)
            self._copy_buffer(self.scaler_std, std)

            X_scaled = (X - mean) / std
            X_normalized = X_scaled / torch.linalg.norm(X_scaled, dim=1, keepdim=True).clamp_min(1e-8)

            n_components = min(self.output_dim, X_normalized.shape[1], X_normalized.shape[0])
            _, _, v = torch.pca_lowrank(X_normalized, q=n_components, center=False)
            components = v[:, :n_components].T.contiguous()

            padded = torch.zeros(self.output_dim, self.input_dim)
            padded[:components.shape[0], :components.shape[1]] = components
            self._copy_buffer(self.svd_components, padded)

            z_tensor = torch.matmul(X_normalized, padded.t())
            z_tensor = z_tensor / torch.linalg.norm(z_tensor, dim=1, keepdim=True).clamp_min(1e-8)
            Z = z_tensor.numpy()

        # 标记为已拟合
        self.fitted_flag.fill_(True)

        # ==========================================
        result = torch.tensor(Z, dtype=torch.float32)

        return result

    def transform(self, features):
        """转换特征（使用保存在 buffer 中的参数）"""
        if not self.fitted_flag.item():
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # 转换为 numpy（如果需要）
        if torch.is_tensor(features):
            X = features.cpu().numpy()
            device = features.device
        else:
            X = features
            device = torch.device('cpu')

        # 使用保存的 scaler 参数进行标准化
        X_scaled = (X - self.scaler_mean.cpu().numpy()) / (self.scaler_std.cpu().numpy() + 1e-8)

        # L2 归一化
        X_normalized = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)

        # 该模块在 NumPy/CPU 中拟合；投影也放在 CPU，避免不同 CANN 版本
        # 对 SVD/复数算子覆盖不一致。结果再回到调用方的设备。
        svd_components = self.svd_components.detach().cpu().float()
        X_tensor = torch.from_numpy(X_normalized).float()
        Z = torch.matmul(X_tensor, svd_components.t()).numpy()

        # L2 归一化 SVD 输出
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

        return torch.tensor(Z, dtype=torch.float32).to(device)

    @staticmethod
    def _copy_buffer(buffer: torch.Tensor, value: torch.Tensor) -> None:
        buffer.copy_(value.to(device=buffer.device, dtype=buffer.dtype))

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
        device = state.device
        theta = self.theta.to(device=device).reshape(())

        if self.gate_type == 'RY':
            cos_half = torch.cos(theta / 2)
            sin_half = torch.sin(theta / 2)
            gate_matrix = torch.stack(
                [torch.stack([cos_half, -sin_half]), torch.stack([sin_half, cos_half])]
            ).to(device=device, dtype=state.dtype)
        elif self.gate_type == 'RZ':
            exp_pos = torch.exp(1j * theta / 2)
            exp_neg = torch.exp(-1j * theta / 2)
            zero = torch.zeros((), device=device, dtype=state.dtype)
            gate_matrix = torch.stack(
                [torch.stack([exp_neg, zero]), torch.stack([zero, exp_pos])]
            ).to(device=device, dtype=state.dtype)
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
        # identity = torch.eye(2, device=device, dtype=dtype)
        # if self.qubit_idx == 0:
        #     result = gate_matrix
        #     for i in range(1, self.n_qubits):
        #         result = torch.kron(result, identity)
        # else:
        #     result = identity
        #     for i in range(1, self.n_qubits):
        #         if i == self.qubit_idx:
        #             result = torch.kron(result, gate_matrix)
        #         else:
        #             result = torch.kron(result, identity)
        # return result

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
    """量子特征编码器 - 使用角度编码和数据重新上传"""
    def __init__(self, input_dim: int = 12, n_qubits: int = 6, data_reuploading: int = 2):
        super().__init__()
        self.input_dim = input_dim  # 12维输入
        self.n_qubits = n_qubits     # 6个量子比特
        self.data_reuploading = data_reuploading  # 2次数据重新上传
        self.chunk_size = n_qubits   # 每次上传6维

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        将12维特征编码为量子态
        features: [B, 12]
        返回: [B, n_qubits, data_reuploading] 角度参数
        """
        B, D = features.shape
        expected_dim = self.n_qubits * self.data_reuploading

        if D < expected_dim:
            pad = torch.zeros(B, expected_dim - D, device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad], dim=1)
        elif D > expected_dim:
            features = features[:, :expected_dim]

        angles = features.view(B, self.data_reuploading, self.n_qubits).transpose(1, 2)
        angles = angles * math.pi
        return angles

class QuantumTaskIdentifier(nn.Module):
    """
    量子任务识别器 - 使用角度编码 + 数据重新上传 + Pauli-Z 测量 + task_head监督分类
    输入: 12维SVD特征
    量子电路: 6 qubits, depth=3, data-reuploading=2
    读出: 测量3个量子比特的Pauli-Z期望值 → task_head → softmax
    """
    def __init__(self, feature_dim: int = 12, n_qubits: int = 6, n_layers: int = 3,
                 max_tasks: int = 10, data_reuploading: int = 2, n_readout_qubits: int = 3,
                 quantum_device: str = "auto", reference_device=None, svd_random_state: int = 1993):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_tasks = max_tasks
        self.data_reuploading = data_reuploading
        self.n_readout_qubits = min(int(n_readout_qubits), int(n_qubits))
        reference = reference_device if isinstance(reference_device, torch.device) else torch.device(reference_device or "cpu")
        self.quantum_device = resolve_quantum_device(quantum_device, reference)

        if self.feature_dim != self.n_qubits * self.data_reuploading:
            logging.warning(
                "feature_dim (%d) != n_qubits (%d) * data_reuploading (%d), encoder will pad/truncate automatically",
                self.feature_dim,
                self.n_qubits,
                self.data_reuploading,
            )

        # 特征预处理器
        self.preprocessor = FeaturePreprocessor(
            input_dim=768,
            output_dim=feature_dim,
            random_state=svd_random_state,
        )

        # 量子特征编码器
        self.encoder = QuantumFeatureEncoder(
            input_dim=feature_dim,
            n_qubits=n_qubits,
            data_reuploading=data_reuploading,
        )

        # 量子层 - 每层包含RY旋转 + CNOT纠缠
        self.quantum_layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            for reupload_idx in range(data_reuploading):
                layer_gates = nn.ModuleList()
                # RY旋转门（数据编码）
                for qubit in range(n_qubits):
                    layer_gates.append(ParameterizedQuantumGate(n_qubits, qubit, 'RY'))
                # 纠缠层
                for qubit in range(n_qubits - 1):
                    layer_gates.append(CNOTGate(n_qubits, qubit, qubit + 1))
                self.quantum_layers.append(layer_gates)

        # 读出维度（Pauli-Z测量得到的向量维度）
        self.readout_dim = n_readout_qubits

        # 任务分类头：从readout直接预测任务ID（监督分类）
        self.task_head = nn.Linear(self.readout_dim, max_tasks)

        # 任务活跃状态
        self.register_buffer('task_active', torch.zeros(max_tasks, dtype=torch.bool))
        self.current_task_id = -1

        # ============ PQK: 存储每个任务的原型约化密度矩阵 ============
        # 每个任务的约化密度矩阵 (2x2, 针对qubit 0)
        self.task_reduced_density_matrices = {}  # {task_id: tensor[2, 2]}
        self.pqk_temperature = 1.0  # softmax 温度参数

    def set_task_id(self, task_id: int):
        """设置当前任务ID"""
        self.current_task_id = task_id
        self.task_active[task_id] = True

    def _compute_reduced_density_matrix(self, quantum_state: torch.Tensor, target_qubit: int = 0):
        """
        计算指定qubit的1-qubit约化密度矩阵
        ρ^(1)(x) = Tr_{j≠target}(|ϕ(x)⟩⟨ϕ(x)|)

        Args:
            quantum_state: [B, 2^n] 完整量子态
            target_qubit: 目标量子比特索引 (默认为0)
        Returns:
            rho: [B, 2, 2] 约化密度矩阵
        """
        B = quantum_state.shape[0]
        state_dim = quantum_state.shape[1]
        n_qubits = int(math.log2(state_dim))

        # 初始化约化密度矩阵 [B, 2, 2]
        rho = torch.zeros(B, 2, 2, dtype=torch.complex64, device=quantum_state.device)

        # 对每个基态计算贡献
        for basis_state in range(state_dim):
            # 提取target_qubit的比特值
            target_bit = (basis_state >> (n_qubits - 1 - target_qubit)) & 1

            for basis_state_prime in range(state_dim):
                target_bit_prime = (basis_state_prime >> (n_qubits - 1 - target_qubit)) & 1

                # 检查其他qubit是否相同（trace out其他qubits）
                other_bits_same = True
                for q in range(n_qubits):
                    if q != target_qubit:
                        bit = (basis_state >> (n_qubits - 1 - q)) & 1
                        bit_prime = (basis_state_prime >> (n_qubits - 1 - q)) & 1
                        if bit != bit_prime:
                            other_bits_same = False
                            break

                if other_bits_same:
                    # ρ[i,j] += ψ[...i...] * ψ*[...j...]
                    rho[:, target_bit, target_bit_prime] += quantum_state[:, basis_state] * torch.conj(quantum_state[:, basis_state_prime])

        return rho

    def _compute_pqk_scores(self, current_rho: torch.Tensor):
        """
        计算当前样本与所有任务原型的PQK核分数
        w_t = tr(ρ^(1)(x) ρ^(1)_t)

        Args:
            current_rho: [B, 2, 2] 当前样本的约化密度矩阵
        Returns:
            pqk_scores: [B, num_tasks] PQK核分数
            alpha_weights: [B, num_tasks] softmax后的权重
        """
        B = current_rho.shape[0]
        num_tasks = len(self.task_reduced_density_matrices)

        if num_tasks == 0:
            # 没有历史任务，返回空
            return None, None

        # 计算与每个任务的核分数
        pqk_scores = torch.zeros(B, num_tasks, device=current_rho.device)

        for task_id in sorted(self.task_reduced_density_matrices.keys()):
            task_rho = self.task_reduced_density_matrices[task_id].to(current_rho.device)  # [2, 2]

            # 计算 tr(ρ(x) * ρ_t) for each sample
            # [B, 2, 2] @ [2, 2] -> [B, 2, 2] -> trace -> [B]
            product = torch.matmul(current_rho, task_rho.unsqueeze(0))  # [B, 2, 2]
            trace_val = torch.diagonal(product, dim1=-2, dim2=-1).sum(dim=-1)  # [B]
            pqk_scores[:, task_id] = trace_val.real  # 取实部

        # ==========================================

        # 应用softmax获得权重
        alpha_weights = F.softmax(pqk_scores / self.pqk_temperature, dim=-1)

        return pqk_scores, alpha_weights

    def save_task_prototype(self, task_id: int, features: torch.Tensor):
        """
        保存任务的原型约化密度矩阵

        Args:
            task_id: 任务ID
            features: [N, 768] 该任务的所有训练样本特征
        """
        with torch.no_grad():
            # 预处理特征
            z_batch = self.preprocessor.transform(features).to(features.device)
            angles = self.encoder(z_batch)

            # 通过量子电路
            quantum_state = self._run_quantum_circuit(z_batch, angles)

            # 计算约化密度矩阵
            rho = self._compute_reduced_density_matrix(quantum_state, target_qubit=0)  # [N, 2, 2]

            # 平均所有样本得到原型
            prototype_rho = rho.mean(dim=0)  # [2, 2]

            # 保存
            self.task_reduced_density_matrices[task_id] = prototype_rho.cpu()

    def save_prototypes_to_file(self, filepath: str):
        """保存所有任务原型到文件"""
        prototypes = {
            task_id: rho.detach().cpu().numpy()
            for task_id, rho in self.task_reduced_density_matrices.items()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(prototypes, f)
        print(f"[INFO] Saved {len(prototypes)} task prototypes to {filepath}")

    def load_prototypes_from_file(self, filepath: str):
        """从文件加载任务原型"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                prototypes = pickle.load(f)
            self.task_reduced_density_matrices = {
                task_id: torch.from_numpy(rho) for task_id, rho in prototypes.items()
            }
            print(f"[INFO] Loaded {len(prototypes)} task prototypes from {filepath}")
        else:
            print(f"[WARN] Prototype file not found: {filepath}")

    def _run_quantum_circuit(self, z_batch: torch.Tensor, angles: torch.Tensor):
        """
        运行量子电路（辅助方法）

        Args:
            z_batch: [B, 12] 预处理后的特征
            angles: [B, 6, 2] 编码角度
        Returns:
            quantum_state: [B, 2^n_qubits] 量子态
        """
        angles = angles.to(self.quantum_device)
        B = angles.shape[0]
        state_dim = 2 ** self.n_qubits
        quantum_state = torch.zeros(
            B, state_dim, dtype=torch.complex64, device=self.quantum_device
        )
        quantum_state[:, 0] = 1.0  # |000000>

        # 通过量子层（与原forward相同的电路）
        layer_idx = 0
        for depth in range(self.n_layers):
            for reupload_idx in range(self.data_reuploading):
                layer_gates = self.quantum_layers[layer_idx]
                gate_idx = 0

                # RY旋转门
                for qubit in range(self.n_qubits):
                    ry_gate = layer_gates[gate_idx]
                    data_angles = angles[:, qubit, reupload_idx]

                    for b in range(B):
                        cos_half = torch.cos(data_angles[b] / 2)
                        sin_half = torch.sin(data_angles[b] / 2)
                        data_gate = torch.stack(
                            [
                                torch.stack([cos_half, -sin_half]),
                                torch.stack([sin_half, cos_half]),
                            ]
                        ).to(device=quantum_state.device, dtype=quantum_state.dtype)

                        ops = [torch.eye(2, device=quantum_state.device, dtype=quantum_state.dtype)
                               for _ in range(self.n_qubits)]
                        ops[qubit] = data_gate
                        full_gate = ops[0]
                        for op in ops[1:]:
                            full_gate = torch.kron(full_gate, op)

                        quantum_state[b:b+1] = torch.matmul(full_gate, quantum_state[b:b+1].unsqueeze(-1)).squeeze(-1)

                    quantum_state = ry_gate(quantum_state)
                    gate_idx += 1

                # CNOT纠缠
                for qubit in range(self.n_qubits - 1):
                    cnot_gate = layer_gates[gate_idx]
                    quantum_state = cnot_gate(quantum_state)
                    gate_idx += 1

                layer_idx += 1

        return quantum_state

    def readout_and_logits(self, features: torch.Tensor):
        """Run the circuit once and return outputs on the caller's device."""
        z_batch = self.preprocessor.transform(features).to(features.device)
        angles = self.encoder(z_batch)
        quantum_state = self._run_quantum_circuit(z_batch, angles)
        readout = self._measure_pauli_z(
            quantum_state, self.n_readout_qubits
        ).to(features.device)
        return readout, self.task_head(readout)

    def forward(self, features, task_id=None, mode='training'):
        """
        前向传播（使用PQK替代Pauli-Z测量）
        Args:
            features: [B, 768] 原始特征
            task_id: 当前任务ID（训练时使用）
            mode: 'training' 或 'inference'
        """
        B = features.shape[0]

        # 预处理：Truncated SVD降维到12维
        if mode == 'training' and not self.preprocessor.fitted_flag.item():
            # 第一次训练时拟合预处理器
            z_batch = self.preprocessor.fit(features).to(features.device)
        else:
            # 使用已拟合的预处理器
            z_batch = self.preprocessor.transform(features).to(features.device)

        # 获取编码角度 [B, 6, 2]
        angles = self.encoder(z_batch)

        # 运行量子电路
        quantum_state = self._run_quantum_circuit(z_batch, angles)

        # ============ PQK: 计算约化密度矩阵 ============
        current_rho = self._compute_reduced_density_matrix(quantum_state, target_qubit=0)  # [B, 2, 2]

        if mode == 'training':
            # 训练模式：计算传统readout用于监督任务分类
            # 为了保持与原有训练接口的兼容性，仍然计算Pauli-Z期望值用于task_head
            readout = self._measure_pauli_z(quantum_state, self.n_readout_qubits).to(features.device)  # [B, 3]
            task_logits = self.task_head(readout)  # [B, max_tasks]

            # 计算PQK权重（用于蒸馏）
            pqk_scores, pqk_weights = self._compute_pqk_scores(current_rho)
            if pqk_weights is not None:
                pqk_weights = pqk_weights.to(features.device)

            return task_logits, readout, pqk_weights
        else:
            # 推理模式：使用PQK进行任务预测
            if len(self.task_reduced_density_matrices) == 0:
                # 没有历史任务，使用传统方法
                readout = self._measure_pauli_z(quantum_state, self.n_readout_qubits).to(features.device)
                task_logits = self.task_head(readout)
                preds = torch.argmax(task_logits, dim=-1)
                confidences = torch.softmax(task_logits, dim=-1)[torch.arange(B), preds]
                return preds.to(features.device), confidences.to(features.device)
            else:
                # 使用PQK预测
                pqk_scores, pqk_weights = self._compute_pqk_scores(current_rho)
                preds = torch.argmax(pqk_weights, dim=-1)
                confidences = torch.max(pqk_weights, dim=-1)[0]
                return preds.to(features.device), confidences.to(features.device)


    def _measure_pauli_z(self, quantum_state: torch.Tensor, n_qubits_to_measure: int):
        """
        测量指定量子比特的Pauli-Z期望值
        Args:
            quantum_state: [B, 2^n] 量子态
            n_qubits_to_measure: 要测量的量子比特数量
        Returns:
            expectation_values: [B, n_qubits_to_measure]
        """
        B = quantum_state.shape[0]
        state_dim = quantum_state.shape[1]
        n_qubits = int(math.log2(state_dim))

        expectations = []
        for qubit_idx in range(n_qubits_to_measure):
            # 计算该量子比特的Pauli-Z期望值
            # Z期望值 = sum_{i: qubit_idx位为0} |psi_i|^2 - sum_{i: qubit_idx位为1} |psi_i|^2
            exp_val = torch.zeros(B, device=quantum_state.device)

            for basis_state in range(state_dim):
                # 检查该基态在qubit_idx位上的值
                bit_value = (basis_state >> (n_qubits - 1 - qubit_idx)) & 1
                prob = torch.abs(quantum_state[:, basis_state]) ** 2

                if bit_value == 0:
                    exp_val += prob
                else:
                    exp_val -= prob

            expectations.append(exp_val)

        return torch.stack(expectations, dim=1)  # [B, n_qubits_to_measure]

    def predict_task(self, features: torch.Tensor, threshold: float = 0.5):
        """预测任务ID（使用task_head）"""
        predicted_tasks, confidences = self.forward(features, task_id=None, mode='inference')

        # 只考虑活跃任务
        if self.task_active.sum() <= 1:
            predicted_tasks.fill_(0)
        else:
            # 低置信度的设为当前最新任务
            unknown_mask = confidences < threshold
            latest_task = torch.sum(self.task_active).item() - 1
            predicted_tasks[unknown_mask] = max(0, latest_task)

        return predicted_tasks, confidences

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

        # ====== Quantum Task Identifier ======
        n_qubits = int(getattr(tuning_config, "n_qubits", 6))
        svd_dim = int(getattr(tuning_config, "svd_dim", 12))
        data_reuploading = int(getattr(tuning_config, "data_reuploading", 2))
        quantum_layers = int(getattr(tuning_config, "quantum_layers", 3))
        self.quantum_identifier = QuantumTaskIdentifier(
            feature_dim=svd_dim,
            n_qubits=n_qubits,
            n_layers=quantum_layers,
            max_tasks=tuning_config.max_tasks,
            data_reuploading=data_reuploading,
            n_readout_qubits=min(3, n_qubits),
            quantum_device=getattr(tuning_config, "quantum_device", "auto"),
            reference_device=getattr(tuning_config, "_device", "cpu"),
            svd_random_state=int(getattr(tuning_config, "svd_random_state", 1993)),
        )

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

    def _compute_confidence_scores(self, adapter_logits):
        """
        计算每个adapter的自信度分数
        Args:
            adapter_logits: [B, num_adapters, num_classes]
        Returns:
            conf_scores: [B, num_adapters] 自信度分数
        """
        B, num_adapters, num_classes = adapter_logits.shape

        # 对每个adapter的logits做softmax
        adapter_probs = F.softmax(adapter_logits, dim=-1)  # [B, num_adapters, num_classes]

        # 对每个adapter，找到最大和次大概率
        sorted_probs, _ = torch.sort(adapter_probs, dim=-1, descending=True)  # [B, num_adapters, num_classes]
        max_prob = sorted_probs[:, :, 0]  # [B, num_adapters]
        second_max_prob = sorted_probs[:, :, 1]  # [B, num_adapters]

        # 自信度 = max - second_max
        conf_scores = max_prob - second_max_prob  # [B, num_adapters]

        return conf_scores

    def _compute_quantum_kd_weights(self, score_logits_q, adapter_logits=None, use_confidence=True, kd_temp=1.0, detach_gate=True):
        """
        使用量子电路输出生成 KD 权重（互信息调制信号）
        Args:
            score_logits_q: tensor [B, K] -- 量子电路输出的 score logits
            adapter_logits: optional tensor [B, K, C] -- adapter logits用于计算置信度
            use_confidence: bool -- 是否结合 adapter confidence
            kd_temp: float -- 温度参数
            detach_gate: bool -- 是否 detach 量子gate以避免反传
        Returns:
            w_final: [B, K] 归一化的 KD 权重
        """
        # 将量子 score 通过温度 softmax -> 基础权重
        w_q = F.softmax(score_logits_q / kd_temp, dim=-1)  # [B, K]
        # 可选: 计算 adapter confidence（基于 logits 的置信度）
        if use_confidence and adapter_logits is not None:
            # adapter_logits: [B, K, C]
            probs = F.softmax(adapter_logits, dim=-1)  # [B, K, C]
            # top1 - top2 confidence
            topk_vals, _ = probs.topk(2, dim=-1)  # [B, K, 2]
            top1 = topk_vals[..., 0]  # [B, K]
            second = topk_vals[..., 1]  # [B, K]
            conf = (top1 - second).clamp(min=0.0)  # [B, K]
            # normalize conf to [0,1]
            conf = conf / (conf.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            conf = None

        # 合成最终权重：w = normalize( w_q * conf ) 或直接 w_q
        if conf is not None:
            if detach_gate:
                w_q_det = w_q.detach()
            else:
                w_q_det = w_q
            w_raw = w_q_det * conf
        else:
            w_raw = w_q

        # renormalize per sample
        w_sum = w_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w_final = w_raw / w_sum
        return w_final
    
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

    def forward_train_with_distillation(self, x):
        """
        训练阶段：使用PQK权重计算蒸馏损失
        返回当前adapter输出、历史adapters输出、以及基于PQK的融合权重
        """
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

        # ============ PQK: 获取PQK权重 ============
        pqk_weights = None
        with torch.no_grad():
            # 调用量子识别器的forward方法（训练模式，但不需要梯度）
            quantum_output = self.quantum_identifier(features_for_quantum, task_id=None, mode='training')

            # quantum_output返回: (task_logits, readout, pqk_weights)
            if len(quantum_output) == 3:
                _, _, pqk_weights = quantum_output  # [B, num_old_adapters] or None
            else:
                # 第一个任务，没有PQK权重
                pqk_weights = None

        # 🎯 通过当前adapter
        x_cur = copy.deepcopy(x_init)
        for idx, blk in enumerate(self.blocks):
            x_cur = blk(x_cur, self.cur_adapter[idx])

        if self.global_pool:
            x_cur = x_cur[:, 1:, :].mean(dim=1)
            outcome_cur = self.fc_norm(x_cur)
        else:
            x_cur = self.norm(x_cur)
            outcome_cur = x_cur[:, 0]

        # 🎯 如果有历史adapters，计算它们的输出
        num_old_adapters = len(self.adapter_list)
        old_adapter_logits = []

        if num_old_adapters > 0:
            with torch.no_grad():
                for task_id in range(num_old_adapters):
                    adapters = self.adapter_list[task_id]

                    x_old = copy.deepcopy(x_init)
                    for i, blk in enumerate(self.blocks):
                        x_old = blk(x_old, adapters[i])

                    x_old = self.norm(x_old)
                    old_feature = x_old[:, 0]

                    # 获取历史adapter的logits
                    old_logits = self.head(old_feature)
                    old_adapter_logits.append(old_logits)

            # 堆叠 [B, num_old_adapters, num_classes]
            old_adapter_logits = torch.stack(old_adapter_logits, dim=1)
        else:
            old_adapter_logits = None

        res = dict()
        res['x'] = outcome_cur
        res['features'] = outcome_cur
        res['old_logits'] = old_adapter_logits  # [B, num_old_adapters, num_classes] or None
        res['gate_weights'] = None  # 不再使用gate_weights
        res['old_conf_scores'] = None  # 不再使用conf_scores
        res['final_weights'] = pqk_weights  # [B, num_old_adapters] or None (PQK权重)
        res['readout'] = None  # 不再需要readout
        return res

    def forward_train_quan(self, x):
        """训练量子识别器时的特征提取（不使用adapter）"""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 不使用任何adapter，直接通过blocks
        for idx, blk in enumerate(self.blocks):
            x = blk(x, adapt=None)

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
        """
        测试阶段：使用量子门控 + adapter自信度加权融合
        """
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

        # 🎯 确定需要测试的adapter数量
        # 在推理阶段，应该测试所有已完成的任务
        # adapter_list 包含所有已完成并保存的adapters
        available_adapters = []

        # 添加所有已完成的历史adapters
        for i, adapter in enumerate(self.adapter_list):
            available_adapters.append((i, adapter, f"task_{i}"))

        # 只有在训练阶段且当前adapter尚未添加到list时，才包含cur_adapter
        # 在推理阶段，所有应该测试的adapter都已经在adapter_list中了
        # 检查是否存在遗漏的当前adapter（通常发生在训练刚完成但还未调用add_adapter_to_list时）
        num_expected_adapters = self.cur_task_id + 1  # 任务ID从0开始，所以总数是cur_task_id+1
        if len(self.adapter_list) < num_expected_adapters and hasattr(self, 'cur_adapter') and self.cur_adapter is not None:
            # 只有当adapter_list中的adapter数量少于预期时，才添加cur_adapter
            available_adapters.append((len(self.adapter_list), self.cur_adapter, f"task_{self.cur_task_id}"))

        num_active_adapters = len(available_adapters)

        if num_active_adapters == 0:
            # 安全回退：至少使用当前adapter
            available_adapters = [(0, self.cur_adapter, "task_current")]
            num_active_adapters = 1

        # 🎯 量子任务识别：获取readout向量和task_logits
        with torch.no_grad():
            readout, task_logits = self.quantum_identifier.readout_and_logits(
                features_for_quantum
            )

            # 量子门控权重：确保使用与实际adapter数量一致的维度
            if readout.shape[1] < num_active_adapters:
                # readout 维度不足，使用task_logits代替
                if task_logits.shape[1] >= num_active_adapters:
                    gate_w = F.softmax(task_logits[:, :num_active_adapters], dim=-1)
                else:
                    # task_logits 维度也不足，使用均匀权重
                    gate_w = torch.ones(B, num_active_adapters, device=features_for_quantum.device) / num_active_adapters
            else:
                gate_w = F.softmax(readout[:, :num_active_adapters], dim=-1)  # [B, num_active_adapters]

        # 🎯 用每个adapter提取特征和logits
        adapter_features = []
        adapter_logits_list = []

        for task_id, adapters, task_name in available_adapters:
            # 通过blocks with adapter
            x_task = copy.deepcopy(x_init)
            for i, blk in enumerate(self.blocks):
                x_task = blk(x_task, adapters[i])

            x_task = self.norm(x_task)
            task_feature = x_task[:, 0]  # [B, embed_dim]

            # 获取该adapter的分类logits
            task_logits_adapter = self.head(task_feature)  # [B, num_classes]

            adapter_features.append(task_feature)
            adapter_logits_list.append(task_logits_adapter)

        # 堆叠 [B, num_adapters, ...]
        adapter_features = torch.stack(adapter_features, dim=1)  # [B, num_active_adapters, embed_dim]
        adapter_logits = torch.stack(adapter_logits_list, dim=1)  # [B, num_active_adapters, num_classes]

        # 🎯 计算每个adapter的自信度分数
        conf_scores = self._compute_confidence_scores(adapter_logits)  # [B, num_active_adapters]

        # 🎯 联合权重：量子门控 × adapter自信度
        # 现在两者的维度应该完全一致：[B, num_active_adapters]
        w_combined = gate_w * conf_scores  # [B, num_active_adapters]
        w_final = F.softmax(w_combined, dim=-1)  # [B, num_active_adapters]

        # 🎯 加权融合特征
        w_final_expanded = w_final.unsqueeze(-1)  # [B, num_active_adapters, 1]
        outcome = torch.sum(adapter_features * w_final_expanded, dim=1)  # [B, embed_dim]

        # 预测任务（取权重最大的）
        predicted_tasks = torch.argmax(w_final, dim=-1)

        res = dict()
        res['x'] = outcome
        res['predicted_tasks'] = predicted_tasks
        res['features'] = outcome
        res['gate_weights'] = gate_w
        res['conf_scores'] = conf_scores
        res['final_weights'] = w_final
        return res

    def forward_test_with_quantum_soft_routing(self, x, temperature: float = 1.0):
        """
        测试阶段：使用量子软路由（加权融合多个adapter的特征）
        Args:
            x: 输入图像 [B, 3, 224, 224]
            temperature: 温度参数，控制任务概率分布的平滑度
        """
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

        # 🎯 量子任务识别：使用 task_head 获取任务概率分布
        with torch.no_grad():
            _, task_logits = self.quantum_identifier.readout_and_logits(
                features_for_quantum
            )

            # 使用temperature缩放的softmax获取任务概率
            task_probs = torch.softmax(task_logits / temperature, dim=-1)

        # 获取活跃任务（限制在有效范围内）
        # 注意：cur_task_id 在训练完一个任务后会递增（指向下一个待训练的任务）
        # 所以实际已完成的任务是 [0, cur_task_id-1]，当前正在训练的任务是 cur_task_id
        active_tasks = []

        # 计算实际已完成的任务数量
        num_finished_tasks = len(self.adapter_list)  # adapter_list中保存的是已完成任务的adapter
        # 只考虑已完成的任务（在adapter_list中的）
        for task_id in range(num_finished_tasks):
            if task_id < self.quantum_identifier.max_tasks:  # 确保不超过task_head的维度
                active_tasks.append(task_id)
        # 检查索引安全性
        if len(active_tasks) > 0 and max(active_tasks) >= task_probs.shape[1]:
            print(f"[ERROR] Index out of bounds: max(active_tasks)={max(active_tasks)}, task_probs.shape[1]={task_probs.shape[1]}")
            # 修复：过滤掉超出范围的任务
            active_tasks = [t for t in active_tasks if t < task_probs.shape[1]]
            print(f"[FIX] Filtered active_tasks={active_tasks}")

        if len(active_tasks) == 0:
            # 安全回退：使用第一个任务（如果是在第一个任务训练期间推理）
            print(f"[WARN] No active tasks found, using task 0 as fallback")
            active_tasks = [0]

        # 用每个活跃任务的adapter分别提取特征
        task_features = []
        for task_id in active_tasks:
            # 从adapter_list中获取对应的adapter
            if task_id < len(self.adapter_list):
                adapters = self.adapter_list[task_id]
            else:
                # 这种情况不应该发生，因为active_tasks已经限制在adapter_list范围内
                print(f"[ERROR] task_id={task_id} >= adapter_list length={len(self.adapter_list)}, using last adapter")
                adapters = self.adapter_list[-1] if len(self.adapter_list) > 0 else self.cur_adapter

            # 通过blocks with selected adapter
            x_task = copy.deepcopy(x_init)
            for i, blk in enumerate(self.blocks):
                x_task = blk(x_task, adapters[i])

            x_task = self.norm(x_task)
            task_features.append(x_task[:, 0])  # [B, embed_dim]

        # 加权融合特征
        task_features = torch.stack(task_features, dim=1)  # [B, num_active_tasks, embed_dim]

        # 🔧 修复索引越界：只取 active_tasks 对应的概率
        # task_probs 的形状是 [B, max_tasks]，我们只取其中活跃任务的列
        task_probs_active = task_probs[:, active_tasks]  # [B, num_active_tasks]
        task_probs_active = task_probs_active.unsqueeze(-1)  # [B, num_active_tasks, 1]

        outcome = torch.sum(task_features * task_probs_active, dim=1)  # [B, embed_dim]

        # 记录预测任务（取概率最大的）
        active_task_indices = torch.argmax(task_probs[:, active_tasks], dim=1)  # [B]，范围 [0, num_active_tasks-1]
        predicted_tasks = torch.tensor([active_tasks[i] for i in active_task_indices.cpu().numpy()],
                                      device=x.device, dtype=torch.long)

        res = dict()
        res['x'] = outcome
        res['predicted_tasks'] = predicted_tasks
        res['task_probs'] = task_probs  # 额外返回任务概率分布
        res['features'] = outcome
        return res

    def forward(self, x, test=False, fc_only=False, quan=False, use_soft_routing=False, temperature=1.0):
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
            if use_soft_routing:
                res = self.forward_test_with_quantum_soft_routing(x, temperature)
            else:
                res = self.forward_test_with_quantum(x)
            res = self.forward_head(res)
            return res

def vit_base_patch16_224_quan_adapter(pretrained=False, **kwargs):
    checkpoint_path = kwargs.pop('checkpoint_path', None) or resolve_checkpoint_path('in1k')
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    ckpt_path = checkpoint_path
    if os.path.exists(ckpt_path):
        checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, checkpoint_path=ckpt_path)
    else:
        logging.warning(f"Checkpoint not found at {ckpt_path}. Falling back to timm pretrained/random init.")
        try:
            checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        except Exception as e:
            logging.warning(f"Failed to load timm pretrained weights: {e}. Using random initialization.")
            checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
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
    checkpoint_path = kwargs.pop('checkpoint_path', None) or resolve_checkpoint_path('in21k')
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    ckpt_path = checkpoint_path
    if os.path.exists(ckpt_path):
        checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=False, num_classes=0, checkpoint_path=ckpt_path)
    else:
        logging.warning(f"Checkpoint not found at {ckpt_path}. Falling back to timm pretrained/random init.")
        try:
            checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=pretrained, num_classes=0)
        except Exception as e:
            logging.warning(f"Failed to load timm pretrained weights: {e}. Using random initialization.")
            checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=False, num_classes=0)
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
