import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import QuanAdapterNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torch.distributions.multivariate_normal import MultivariateNormal
import time
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = QuanAdapterNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"]
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.quantum_epochs = args["quantum_epochs"]

        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n and 'quantum' not in n:
                p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        return start_cls, end_cls

    def get_task_id_from_class(self, class_id):
        if class_id < self.init_cls:
            return 0
        else:
            return 1 + (class_id - self.init_cls) // self.inc

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        # 准备数据
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.data_manager = data_manager
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            logging.info('Using Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # 两阶段训练
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)

        # ========== 性能统计：训练阶段 ==========
        torch.cuda.reset_peak_memory_stats(self._device)
        train_start_time = time.time()

        # 阶段1: 训练adapter
        logging.info(f"Stage 1: Training Adapter for Task {self._cur_task}")
        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)

        adapter_start_time = time.time()
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        adapter_end_time = time.time()
        adapter_time = adapter_end_time - adapter_start_time

        self._network.backbone.add_adapter_to_list()

        # 阶段2: 训练量子识别器
        logging.info(f"Stage 2: Training Quantum Task Identifier for Task {self._cur_task}")
        quantum_start_time = time.time()
        self._train_quantum_identifier(train_loader)
        quantum_end_time = time.time()
        quantum_time = quantum_end_time - quantum_start_time

        self._compute_mean(self._network)
        if self._cur_task > 0:
            self.classifer_align(self._network)

        train_end_time = time.time()
        total_train_time = train_end_time - train_start_time
        peak_memory_train = torch.cuda.max_memory_allocated(self._device) / (1024**3)  # GB

        # 记录训练性能统计
        logging.info("=" * 60)
        logging.info(f"[TRAINING PERFORMANCE] Task {self._cur_task}")
        logging.info(f"  Adapter Training Time: {adapter_time:.2f}s")
        logging.info(f"  Quantum Training Time: {quantum_time:.2f}s")
        logging.info(f"  Total Training Time: {total_train_time:.2f}s")
        logging.info(f"  Peak Memory (Training): {peak_memory_train:.3f} GB")
        logging.info("=" * 60)
    
    def get_optimizer(self, model):
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_fc_params = [p for name, p in model.named_parameters() if 'adapter' not in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.init_lr *0.1, 'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params, 
                momentum=0.9,
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
            )
        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['epochs'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler
    
    def _train_quantum_identifier(self, train_loader):
        # 检查是否有可训练参数
        quantum_params = list(self._network.backbone.quantum_identifier.parameters())
        has_trainable_params = len(quantum_params) > 0 and any(p.requires_grad for p in quantum_params)

        # 设置当前任务ID
        self._network.backbone.quantum_identifier.set_task_id(self._cur_task)

        # 如果没有可训练参数（如cosine/mlp/attention），只进行锚点更新，不训练
        if not has_trainable_params:
            logging.info(f"Task Identifier has no trainable parameters, only updating anchors...")

            # 记录时间
            quantum_start_time = time.time()

            self._network.eval()  # 设置为eval模式
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(self._device)
                    features = self._extract_backbone_features(inputs)['features']

                    # 只更新锚点，不计算损失
                    _ = self._network.backbone.quantum_identifier(
                        features, task_id=self._cur_task, mode='training'
                    )

            quantum_end_time = time.time()
            quantum_time = quantum_end_time - quantum_start_time

            # 获取锚点信息
            task_info = self._network.backbone.quantum_identifier.get_task_info()
            logging.info(f"Anchor update completed in {quantum_time:.2f}s")
            logging.info(f"Task {self._cur_task} anchor info: {task_info}")
            return

        # 如果有可训练参数（如quantum），进行正常训练
        optimizer = optim.Adam(quantum_params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.quantum_epochs, eta_min=self.min_lr
        )

        # 记录每个epoch的时间和吞吐量
        quantum_epoch_times = []
        quantum_epoch_throughputs = []

        # ========== 组件级 Runtime Breakdown ==========
        routing_component_times = {
            'backbone_forward': [],
            'routing_forward': [],
            'loss_computation': [],
            'backward': [],
            'optimizer_step': []
        }

        prog_bar = tqdm(range(self.quantum_epochs), desc="Training Quantum Identifier")
        for epoch in prog_bar:
            epoch_start_time = time.time()
            self._network.train()

            # 在每个epoch开始时重置样本计数
            self._network.backbone.quantum_identifier.reset_epoch_counts(self._cur_task)

            losses = 0.0
            separation_losses = 0.0
            total_samples = 0
            samples_processed = 0

            # 每个epoch的组件时间累计
            epoch_routing_times = {
                'backbone_forward': 0.0,
                'routing_forward': 0.0,
                'loss_computation': 0.0,
                'backward': 0.0,
                'optimizer_step': 0.0
            }

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)

                # Backbone Forward (无梯度)
                t0 = time.time()
                with torch.no_grad():
                    features = self._extract_backbone_features(inputs)['features']
                torch.cuda.synchronize(self._device)
                t1 = time.time()
                epoch_routing_times['backbone_forward'] += (t1 - t0)

                # Routing (Task Identifier) Forward
                t2 = time.time()
                quantum_states = self._network.backbone.quantum_identifier(
                    features, task_id=self._cur_task, mode='training'
                )
                torch.cuda.synchronize(self._device)
                t3 = time.time()
                epoch_routing_times['routing_forward'] += (t3 - t2)

                # Loss Computation
                t4 = time.time()
                intra_task_loss = self._compute_clustering_loss(quantum_states)
                torch.cuda.synchronize(self._device)
                t5 = time.time()
                epoch_routing_times['loss_computation'] += (t5 - t4)

                # Backward
                t6 = time.time()
                optimizer.zero_grad()
                intra_task_loss.backward()
                torch.cuda.synchronize(self._device)
                t7 = time.time()
                epoch_routing_times['backward'] += (t7 - t6)

                # Optimizer Step
                t8 = time.time()
                optimizer.step()
                torch.cuda.synchronize(self._device)
                t9 = time.time()
                epoch_routing_times['optimizer_step'] += (t9 - t8)

                # EMA参数更新 - 在每次训练步骤后
                self._network.backbone.quantum_identifier.update_ema()

                losses += intra_task_loss.item()
                total_samples += inputs.size(0)
                samples_processed += inputs.size(0)

                # 记录分离损失（仅用于监控）
                if self._cur_task > 0:
                    with torch.no_grad():
                        sep_loss = self._compute_anchor_separation_loss(quantum_states)
                        separation_losses += sep_loss.item()

            # 记录每个epoch的组件时间
            for key in routing_component_times:
                routing_component_times[key].append(epoch_routing_times[key])

            scheduler.step()

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_throughput = samples_processed / epoch_time

            quantum_epoch_times.append(epoch_time)
            quantum_epoch_throughputs.append(epoch_throughput)

            avg_loss = losses / len(train_loader)
            avg_separation = separation_losses / len(train_loader) if self._cur_task > 0 else 0.0

            # 获取锚点信息
            task_info = self._network.backbone.quantum_identifier.get_task_info()
            current_task_info = task_info.get(f'task_{self._cur_task}', {'active_anchors': 0, 'total_samples': 0})

            info = f"Quantum Epoch {epoch + 1}/{self.quantum_epochs} => Loss {avg_loss:.3f}, Sep {avg_separation:.3f}, Anchors {current_task_info['active_anchors']}, Samples {current_task_info['total_samples']}, Time {epoch_time:.2f}s, Throughput {epoch_throughput:.2f} imgs/s"
            prog_bar.set_description(info)

        logging.info(info)
        logging.info(f"Task {self._cur_task} final anchor info: {task_info}")

        # 记录平均量子训练性能
        avg_quantum_epoch_time = np.mean(quantum_epoch_times)
        avg_quantum_epoch_throughput = np.mean(quantum_epoch_throughputs)
        logging.info(f"[QUANTUM TRAINING] Avg Epoch Time: {avg_quantum_epoch_time:.2f}s, Avg Throughput: {avg_quantum_epoch_throughput:.2f} imgs/s")

        # ========== 组件级 Runtime Breakdown 统计 ==========
        logging.info("=" * 80)
        logging.info(f"[COMPONENT-LEVEL RUNTIME BREAKDOWN] Task {self._cur_task} - Routing Training")
        logging.info("-" * 80)
        total_routing_time = 0.0
        for component, times in routing_component_times.items():
            avg_time = np.mean(times)
            total_routing_time += avg_time
            logging.info(f"  {component:25s}: {avg_time:8.3f}s per epoch")

        logging.info("-" * 80)
        for component, times in routing_component_times.items():
            avg_time = np.mean(times)
            percentage = (avg_time / total_routing_time * 100) if total_routing_time > 0 else 0
            logging.info(f"  {component:25s}: {percentage:6.2f}%")
        logging.info("=" * 80)

        # 任务完成后，重新初始化EMA参数以反映当前状态
        self._network.backbone.quantum_identifier._init_ema_params()
        logging.info(f"EMA parameters reinitialized for Task {self._cur_task}")

    def _compute_clustering_loss(self, quantum_states):
        """计算聚类损失 - 鼓励量子态分布紧凑 + 任务间锚点分离"""
        # 检查是否为复数（量子态）还是实数（其他方法）
        if torch.is_complex(quantum_states):
            # 将复数量子态转换为实数表示（使用概率幅度）
            # 在量子计算中，我们关心的是 |ψ|² 概率分布
            quantum_probs = torch.abs(quantum_states) ** 2  # [B, state_dim]
        else:
            # 对于非量子方法，直接使用特征
            quantum_probs = quantum_states

        # 1. 计算批次内的方差损失，鼓励紧凑聚类
        batch_mean = torch.mean(quantum_probs, dim=0)
        distances = torch.norm(quantum_probs - batch_mean.unsqueeze(0), dim=-1)
        variance_loss = torch.mean(distances)

        # 2. 正则化损失：防止量子态塌缩到同一点
        pairwise_distances = torch.cdist(quantum_probs, quantum_probs, p=2)
        mask = torch.eye(quantum_probs.size(0), device=quantum_probs.device).bool()
        pairwise_distances = pairwise_distances.masked_fill(mask, float('inf'))
        min_distance = torch.min(pairwise_distances)
        diversity_loss = torch.max(
            torch.tensor(0.1, device=quantum_probs.device) - min_distance,
            torch.tensor(0.0, device=quantum_probs.device)
        )

        # 3. 任务间锚点分离损失 - 防止灾难性遗忘
        separation_loss = self._compute_anchor_separation_loss(quantum_states)

        # 总损失：平衡聚类紧凑性、多样性和任务分离
        total_loss = variance_loss + 0.1 * diversity_loss + 0.5 * separation_loss

        return total_loss

    def _compute_anchor_separation_loss(self, current_quantum_states):
        """计算任务间锚点分离损失"""
        if self._cur_task == 0:
            # 第一个任务没有需要分离的旧锚点
            return torch.tensor(0.0, device=current_quantum_states.device)

        quantum_identifier = self._network.backbone.quantum_identifier
        separation_loss = torch.tensor(0.0, device=current_quantum_states.device)

        # 检查是否为复数（量子态）还是实数（其他方法）
        if torch.is_complex(current_quantum_states):
            # 将当前量子态转换为概率分布
            current_probs = torch.abs(current_quantum_states) ** 2
        else:
            current_probs = current_quantum_states

        # 计算与所有旧任务锚点的分离损失
        for old_task_id in range(self._cur_task):
            if not quantum_identifier.task_active[old_task_id]:
                continue

            # 获取旧任务的锚点
            for anchor_idx in range(quantum_identifier.anchors_per_task):
                if quantum_identifier.anchor_counts[old_task_id, anchor_idx] > 0:
                    old_anchor = quantum_identifier.task_anchors[old_task_id, anchor_idx]

                    # 检查旧锚点是否为复数
                    if torch.is_complex(old_anchor):
                        old_anchor_probs = torch.abs(old_anchor) ** 2  # [state_dim]
                    else:
                        old_anchor_probs = old_anchor

                    # 计算当前样本与旧锚点的距离
                    # 我们希望最小化相似度，即最大化距离
                    batch_size = current_probs.size(0)
                    old_anchor_expanded = old_anchor_probs.unsqueeze(0).expand(batch_size, -1)

                    # 使用负数的余弦相似度作为分离损失
                    # 余弦相似度越高，分离损失越大
                    similarities = torch.cosine_similarity(current_probs, old_anchor_expanded, dim=1)

                    # 将相似度转换为分离损失：相似度高时损失大
                    anchor_separation = torch.mean(torch.clamp(similarities, min=0.0))
                    separation_loss += anchor_separation

        return separation_loss

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        epochs = self.args["epochs"]
        prog_bar = tqdm(range(epochs))

        # 记录每个epoch的时间和吞吐量
        epoch_times = []
        epoch_throughputs = []

        # ========== 组件级 Runtime Breakdown ==========
        component_times = {
            'backbone_forward': [],
            'adapter_forward': [],
            'loss_computation': [],
            'backward': [],
            'optimizer_step': []
        }

        for _, epoch in enumerate(prog_bar):
            epoch_start_time = time.time()
            self._network.backbone.train()

            losses = 0.0
            correct, total = 0, 0
            samples_processed = 0

            # 每个epoch的组件时间累计
            epoch_component_times = {
                'backbone_forward': 0.0,
                'adapter_forward': 0.0,
                'loss_computation': 0.0,
                'backward': 0.0,
                'optimizer_step': 0.0
            }

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Backbone + Adapter Forward
                t0 = time.time()
                outputs = self._network(inputs, test=False)
                torch.cuda.synchronize(self._device)
                t1 = time.time()
                epoch_component_times['adapter_forward'] += (t1 - t0)

                # Loss Computation
                t2 = time.time()
                logits = outputs["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')
                loss = F.cross_entropy(logits, targets.long())
                torch.cuda.synchronize(self._device)
                t3 = time.time()
                epoch_component_times['loss_computation'] += (t3 - t2)

                # Backward
                t4 = time.time()
                optimizer.zero_grad()
                loss.backward()
                torch.cuda.synchronize(self._device)
                t5 = time.time()
                epoch_component_times['backward'] += (t5 - t4)

                # Optimizer Step
                t6 = time.time()
                optimizer.step()
                torch.cuda.synchronize(self._device)
                t7 = time.time()
                epoch_component_times['optimizer_step'] += (t7 - t6)

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                samples_processed += inputs.size(0)

            # 记录每个epoch的组件时间
            for key in component_times:
                component_times[key].append(epoch_component_times[key])

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_throughput = samples_processed / epoch_time

            epoch_times.append(epoch_time)
            epoch_throughputs.append(epoch_throughput)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Time {:.2f}s, Throughput {:.2f} imgs/s".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    epoch_time,
                    epoch_throughput,
                )

            prog_bar.set_description(info)

        logging.info(info)

        # 记录平均epoch性能
        avg_epoch_time = np.mean(epoch_times)
        avg_epoch_throughput = np.mean(epoch_throughputs)
        logging.info(f"[ADAPTER TRAINING] Avg Epoch Time: {avg_epoch_time:.2f}s, Avg Throughput: {avg_epoch_throughput:.2f} imgs/s")

        # ========== 组件级 Runtime Breakdown 统计 ==========
        logging.info("=" * 80)
        logging.info(f"[COMPONENT-LEVEL RUNTIME BREAKDOWN] Task {self._cur_task} - Adapter Training")
        logging.info("-" * 80)
        total_component_time = 0.0
        for component, times in component_times.items():
            avg_time = np.mean(times)
            total_component_time += avg_time
            logging.info(f"  {component:25s}: {avg_time:8.3f}s per epoch")

        logging.info("-" * 80)
        for component, times in component_times.items():
            avg_time = np.mean(times)
            percentage = (avg_time / total_component_time * 100) if total_component_time > 0 else 0
            logging.info(f"  {component:25s}: {percentage:6.2f}%")
        logging.info("=" * 80)

    def _extract_backbone_features(self, x):
        """提取backbone特征（使用第一个adapter）"""
        # 简化版本：直接通过当前backbone
        # 在实际实现中，这里应该调用backbone的无adapter版本
        return self._network.backbone.forward(x, test=False, quan=True)

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
        self._network.eval()
        y_pred, y_true = [], []

        # ========== 性能统计：推理阶段 ==========
        # 准备性能测试数据
        warmup_iterations = 20
        test_iterations = 200
        batch_1_times = []
        batch_bs_times = []

        # 获取一个batch用于性能测试
        test_batch = None
        for _, (_, inputs, targets) in enumerate(loader):
            test_batch = inputs.to(self._device)
            break

        if test_batch is not None:
            # Warm-up (batch=1)
            single_input = test_batch[0:1]
            for _ in range(warmup_iterations):
                with torch.no_grad():
                    _ = self._network(single_input, test=True)

            # 测试 batch=1 的 latency
            torch.cuda.synchronize(self._device)
            for _ in range(test_iterations):
                start = time.time()
                with torch.no_grad():
                    _ = self._network(single_input, test=True)
                torch.cuda.synchronize(self._device)
                batch_1_times.append(time.time() - start)

            # Warm-up (batch=bs)
            for _ in range(warmup_iterations):
                with torch.no_grad():
                    _ = self._network(test_batch, test=True)

            # 测试 batch=bs 的 latency
            torch.cuda.synchronize(self._device)
            for _ in range(test_iterations):
                start = time.time()
                with torch.no_grad():
                    _ = self._network(test_batch, test=True)
                torch.cuda.synchronize(self._device)
                batch_bs_times.append(time.time() - start)

            # 计算统计量
            avg_latency_batch1 = np.mean(batch_1_times) * 1000  # ms
            avg_latency_batchbs = np.mean(batch_bs_times) * 1000  # ms
            throughput_batch1 = 1.0 / np.mean(batch_1_times)  # images/sec
            throughput_batchbs = test_batch.size(0) / np.mean(batch_bs_times)  # images/sec

            # 测量推理峰值显存
            torch.cuda.reset_peak_memory_stats(self._device)
            with torch.no_grad():
                _ = self._network(test_batch, test=True)
            peak_memory_inference = torch.cuda.max_memory_allocated(self._device) / (1024**3)  # GB

            # 记录推理性能统计
            logging.info("=" * 60)
            logging.info(f"[INFERENCE PERFORMANCE] Task {self._cur_task}")
            logging.info(f"  Latency (batch=1): {avg_latency_batch1:.3f} ms/image")
            logging.info(f"  Latency (batch={test_batch.size(0)}): {avg_latency_batchbs:.3f} ms/batch")
            logging.info(f"  Throughput (batch=1): {throughput_batch1:.2f} images/sec")
            logging.info(f"  Throughput (batch={test_batch.size(0)}): {throughput_batchbs:.2f} images/sec")
            logging.info(f"  Peak Memory (Inference): {peak_memory_inference:.3f} GB")
            logging.info("=" * 60)

        # 正常的评估流程
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                outputs = self._network(inputs, test=True)
                logits = outputs['logits']
                predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())
                if calc_task_acc:
                    task_ids = (targets - self.init_cls) // self.inc + 1
                    task_logits = torch.zeros(logits.shape).to(self._device)
                    for i, task_id in enumerate(task_ids):
                        if task_id == 0:
                            start_cls = 0
                            end_cls = self.init_cls
                        else:
                            start_cls = self.init_cls + (task_id-1)*self.inc
                            end_cls = self.init_cls + task_id*self.inc
                        task_logits[i, start_cls:end_cls] += logits[i, start_cls:end_cls]
                    # calculate the accuracy of task_id
                    pred_task_ids = (torch.max(logits, dim=1)[1] - self.init_cls) // self.inc + 1
                    task_correct += (pred_task_ids.cpu() == task_ids).sum()

                    pred_task_y = torch.max(task_logits, dim=1)[1]
                    task_acc += (pred_task_y.cpu() == targets).sum()
                    total += len(targets)



        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
        return np.concatenate(y_pred), np.concatenate(y_true)
    
    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size*3, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = model(_inputs.to(self._device), test=False)["features"]

                # 检查特征是否有效
                if torch.isnan(_vectors).any() or torch.isinf(_vectors).any():
                    logging.warning(f"[_compute_mean] Class {class_idx}: features contain NaN/Inf, skipping batch")
                    continue

                vectors.append(_vectors)

            if len(vectors) == 0:
                logging.error(f"[_compute_mean] Class {class_idx}: No valid features, cannot compute mean/cov")
                # 使用随机初始化作为fallback
                feature_dim = model.backbone.embed_dim
                self.cls_mean[class_idx] = torch.randn(feature_dim).to(self._device) * 0.01
                if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                    self.cls_cov[class_idx] = torch.eye(feature_dim).to(self._device)
                continue

            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                mean = features_per_cls.mean(dim=0)

                # 检查均值是否有效
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    logging.error(f"[_compute_mean] Class {class_idx}: computed mean contains NaN/Inf")
                    mean = torch.zeros_like(mean)

                self.cls_mean[class_idx] = mean.to(self._device)

                # 计算协方差并添加正则化
                try:
                    cov = torch.cov(features_per_cls.T) + (torch.eye(mean.shape[-1]) * 1e-3).to(self._device)

                    # 检查协方差是否有效
                    if torch.isnan(cov).any() or torch.isinf(cov).any():
                        logging.error(f"[_compute_mean] Class {class_idx}: computed cov contains NaN/Inf, using identity")
                        cov = torch.eye(mean.shape[-1]).to(self._device)

                    self.cls_cov[class_idx] = cov
                except Exception as e:
                    logging.error(f"[_compute_mean] Class {class_idx}: cov computation failed: {e}, using identity")
                    self.cls_cov[class_idx] = torch.eye(mean.shape[-1]).to(self._device)

            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                mean = features_per_cls.mean(dim=0)

                # 检查均值是否有效
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    logging.error(f"[_compute_mean] Class {class_idx}: computed mean contains NaN/Inf")
                    mean = torch.zeros_like(mean)

                self.cls_mean[class_idx] = mean.to(self._device)

                # 计算方差并添加正则化
                try:
                    cov = torch.cov(features_per_cls.T) + (torch.eye(mean.shape[-1]) * 1e-3).to(self._device)
                    var = torch.diag(cov)

                    # 检查方差是否有效
                    if torch.isnan(var).any() or torch.isinf(var).any():
                        logging.error(f"[_compute_mean] Class {class_idx}: computed var contains NaN/Inf, using ones")
                        var = torch.ones(mean.shape[-1]).to(self._device)

                    self.cls_cov[class_idx] = var
                except Exception as e:
                    logging.error(f"[_compute_mean] Class {class_idx}: var computation failed: {e}, using ones")
                    self.cls_cov[class_idx] = torch.ones(mean.shape[-1]).to(self._device)

            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = self.args["n_centroids"]  # 10
                features_per_cls = vectors.cpu().numpy()

                try:
                    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
                    kmeans.fit(features_per_cls)
                    cluster_labels = kmeans.labels_
                    cluster_means = []
                    cluster_vars = []

                    for i in range(n_clusters):
                        cluster_data = features_per_cls[cluster_labels == i]

                        if len(cluster_data) == 0:
                            logging.warning(f"[_compute_mean] Class {class_idx} cluster {i}: empty cluster, skipping")
                            continue

                        cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                        cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(self._device)

                        # 检查是否有效
                        if torch.isnan(cluster_mean).any() or torch.isinf(cluster_mean).any():
                            logging.warning(f"[_compute_mean] Class {class_idx} cluster {i}: mean contains NaN/Inf, skipping")
                            continue

                        if torch.isnan(cluster_var).any() or torch.isinf(cluster_var).any():
                            logging.warning(f"[_compute_mean] Class {class_idx} cluster {i}: var contains NaN/Inf, skipping")
                            continue

                        # 添加小的正则化项避免零方差
                        cluster_var = cluster_var + 1e-6

                        cluster_means.append(cluster_mean)
                        cluster_vars.append(cluster_var)

                    if len(cluster_means) == 0:
                        logging.error(f"[_compute_mean] Class {class_idx}: all clusters invalid, using fallback")
                        # Fallback: 使用整体均值和方差
                        cluster_means = [torch.tensor(np.mean(features_per_cls, axis=0), dtype=torch.float64).to(self._device)]
                        cluster_vars = [torch.tensor(np.var(features_per_cls, axis=0), dtype=torch.float64).to(self._device) + 1e-6]

                    self.cls_mean[class_idx] = cluster_means
                    self.cls_cov[class_idx] = cluster_vars

                except Exception as e:
                    logging.error(f"[_compute_mean] Class {class_idx}: clustering failed: {e}, using fallback")
                    # Fallback: 使用整体均值和方差
                    cluster_means = [torch.tensor(np.mean(features_per_cls, axis=0), dtype=torch.float64).to(self._device)]
                    cluster_vars = [torch.tensor(np.var(features_per_cls, axis=0), dtype=torch.float64).to(self._device) + 1e-6]
                    self.cls_mean[class_idx] = cluster_means
                    self.cls_cov[class_idx] = cluster_vars


    def classifer_align(self, model):
        model.train()

        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        # 记录分类器对齐的时间
        ca_epoch_times = []

        # ========== 组件级 Runtime Breakdown ==========
        kd_component_times = {
            'data_sampling': [],
            'forward': [],
            'kd_loss': [],
            'backward': [],
            'optimizer_step': []
        }

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:
            epoch_start_time = time.time()

            # 每个epoch的组件时间累计
            epoch_kd_times = {
                'data_sampling': 0.0,
                'forward': 0.0,
                'kd_loss': 0.0,
                'backward': 0.0,
                'optimizer_step': 0.0
            }

            # Data Sampling (Knowledge Distillation)
            t0 = time.time()
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)

                    # 检查均值是否有效
                    if torch.isnan(mean).any() or torch.isinf(mean).any():
                        logging.warning(f"[CA] Class {class_idx}: mean contains NaN/Inf, skipping")
                        continue

                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)

                    # 检查协方差是否有效
                    if torch.isnan(cov).any() or torch.isinf(cov).any():
                        logging.warning(f"[CA] Class {class_idx}: cov contains NaN/Inf, skipping")
                        continue

                    # 添加正则化确保协方差矩阵正定
                    cov = cov + torch.eye(cov.shape[0]).to(self._device) * 1e-3

                    try:
                        m = MultivariateNormal(mean.float(), cov.float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

                        # 检查采样数据是否有效
                        if torch.isnan(sampled_data_single).any() or torch.isinf(sampled_data_single).any():
                            logging.warning(f"[CA] Class {class_idx}: sampled data contains NaN/Inf, skipping")
                            continue

                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
                    except Exception as e:
                        logging.warning(f"[CA] Class {class_idx}: sampling failed with error {e}, skipping")
                        continue

            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]

                        # 检查均值和方差是否有效
                        if torch.isnan(mean).any() or torch.isinf(mean).any():
                            logging.warning(f"[CA] Class {class_idx} cluster {cluster}: mean contains NaN/Inf, skipping")
                            continue

                        if torch.isnan(var).any() or torch.isinf(var).any():
                            logging.warning(f"[CA] Class {class_idx} cluster {cluster}: var contains NaN/Inf, skipping")
                            continue

                        if var.mean() == 0:
                            continue

                        # 增强正则化
                        try:
                            m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-3 * torch.eye(mean.shape[0]).to(mean.device)).float())
                            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

                            # 检查采样数据是否有效
                            if torch.isnan(sampled_data_single).any() or torch.isinf(sampled_data_single).any():
                                logging.warning(f"[CA] Class {class_idx} cluster {cluster}: sampled data contains NaN/Inf, skipping")
                                continue

                            sampled_data.append(sampled_data_single)
                            sampled_label.extend([class_idx] * num_sampled_pcls)
                        except Exception as e:
                            logging.warning(f"[CA] Class {class_idx} cluster {cluster}: sampling failed with error {e}, skipping")
                            continue
            else:
                raise NotImplementedError

            # 检查是否有有效的采样数据
            if len(sampled_data) == 0:
                logging.warning(f"[CA] Epoch {epoch}: No valid sampled data, skipping this epoch")
                continue

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            torch.cuda.synchronize(self._device)
            t1 = time.time()
            epoch_kd_times['data_sampling'] += (t1 - t0)

            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            num_valid_batches = 0

            # 动态计算batch数量
            num_batches = inputs.size(0) // num_sampled_pcls

            for _iter in range(num_batches):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                # 再次检查输入是否有效
                if torch.isnan(inp).any() or torch.isinf(inp).any():
                    logging.warning(f"[CA] Epoch {epoch} batch {_iter}: input contains NaN/Inf, skipping")
                    continue

                # Forward
                t2 = time.time()
                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]
                torch.cuda.synchronize(self._device)
                t3 = time.time()
                epoch_kd_times['forward'] += (t3 - t2)

                # KD Loss Computation
                t4 = time.time()
                loss = F.cross_entropy(logits, tgt)
                torch.cuda.synchronize(self._device)
                t5 = time.time()
                epoch_kd_times['kd_loss'] += (t5 - t4)

                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"[CA] Epoch {epoch} batch {_iter}: loss is NaN/Inf, skipping")
                    continue

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                # Backward
                t6 = time.time()
                optimizer.zero_grad()
                loss.backward()
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                torch.cuda.synchronize(self._device)
                t7 = time.time()
                epoch_kd_times['backward'] += (t7 - t6)

                # Optimizer Step
                t8 = time.time()
                optimizer.step()
                torch.cuda.synchronize(self._device)
                t9 = time.time()
                epoch_kd_times['optimizer_step'] += (t9 - t8)

                losses += loss.item()
                num_valid_batches += 1

            # 记录每个epoch的组件时间
            for key in kd_component_times:
                kd_component_times[key].append(epoch_kd_times[key])

            scheduler.step()

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            ca_epoch_times.append(epoch_time)

            # 避免除以0
            if num_valid_batches > 0 and total > 0:
                ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
                avg_loss = losses / num_valid_batches
            else:
                ca_acc = 0.0
                avg_loss = 0.0

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}, Time {:.2f}s".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                avg_loss,
                ca_acc,
                epoch_time,
            )
            prog_bar.set_description(info)

        logging.info(info)

        # 记录平均分类器对齐性能
        if len(ca_epoch_times) > 0:
            avg_ca_epoch_time = np.mean(ca_epoch_times)
            logging.info(f"[CLASSIFIER ALIGNMENT] Avg Epoch Time: {avg_ca_epoch_time:.2f}s")

        # ========== 组件级 Runtime Breakdown 统计 ==========
        if len(ca_epoch_times) > 0:
            logging.info("=" * 80)
            logging.info(f"[COMPONENT-LEVEL RUNTIME BREAKDOWN] Task {self._cur_task} - KD Training")
            logging.info("-" * 80)
            total_kd_time = 0.0
            for component, times in kd_component_times.items():
                if len(times) > 0:
                    avg_time = np.mean(times)
                    total_kd_time += avg_time
                    logging.info(f"  {component:25s}: {avg_time:8.3f}s per epoch")

            logging.info("-" * 80)
            for component, times in kd_component_times.items():
                if len(times) > 0:
                    avg_time = np.mean(times)
                    percentage = (avg_time / total_kd_time * 100) if total_kd_time > 0 else 0
                    logging.info(f"  {component:25s}: {percentage:6.2f}%")
            logging.info("=" * 80)
