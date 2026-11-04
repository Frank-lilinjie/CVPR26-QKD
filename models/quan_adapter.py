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
from utils.device import loader_kwargs, synchronize
from torch.distributions.multivariate_normal import MultivariateNormal

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = QuanAdapterNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self._ca_device = torch.device("cpu") if self._device.type == "npu" else self._device
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

        # 🎯 蒸馏相关超参数
        self.lambda_kd = args.get("lambda_kd", 1.0)
        self.lambda_sparse = args.get("lambda_sparse", 0.05)
        self.kd_temperature = args.get("kd_temperature", 1.0)
        self.use_distillation = args.get("use_distillation", True)

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
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **loader_kwargs(self.args, self._device),
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **loader_kwargs(self.args, self._device),
        )

        # 两阶段训练
        self._train(self.train_loader, self.test_loader)
        

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)
        # 阶段1: 训练adapter
        logging.info(f"Stage 1: Training Adapter for Task {self._cur_task}")
        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)

        self._init_train(train_loader, test_loader, optimizer, scheduler)

        # 阶段2: 训练量子识别器（在add_adapter_to_list之前）
        logging.info(f"Stage 2: Training Quantum Task Identifier for Task {self._cur_task}")
        self._train_quantum_identifier(train_loader)

        # ============ PQK: 保存任务原型约化密度矩阵 ============
        logging.info(f"[PQK] Computing and saving task prototype for task {self._cur_task}")
        self._save_task_prototype(train_loader)

        # 训练完成后，将adapter添加到列表（这会让cur_task_id递增）
        self._network.backbone.add_adapter_to_list()

        self._compute_mean(self._network)
        if self._cur_task > 0:
            self.classifer_align(self._network)

    def _save_task_prototype(self, train_loader):
        """保存当前任务的原型约化密度矩阵"""
        self._network.eval()
        all_features = []

        with torch.no_grad():
            for _, inputs, _ in train_loader:
                inputs = inputs.to(self._device)
                # 提取backbone特征
                features = self._extract_backbone_features(inputs)['features']  # [B, 768]
                all_features.append(features)

        # 合并所有特征
        all_features = torch.cat(all_features, dim=0)  # [N, 768]

        # 调用量子识别器保存原型
        self._network.backbone.quantum_identifier.save_task_prototype(
            task_id=self._cur_task,
            features=all_features
        )
    
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
        """训练量子识别器 - 纯监督分类（使用task_head）"""

        # 获取量子路由器的所有可训练参数
        quantum_params = []
        for name, param in self._network.backbone.quantum_identifier.named_parameters():
            if param.requires_grad:
                quantum_params.append(param)

        optimizer = optim.Adam(quantum_params, lr=self.init_lr * 0.1, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.quantum_epochs, eta_min=self.min_lr
        )

        # 设置当前任务ID
        self._network.backbone.quantum_identifier.set_task_id(self._cur_task)

        prog_bar = tqdm(range(self.quantum_epochs), desc="Training Quantum Router with task_head")
        for epoch in prog_bar:
            self._network.train()

            epoch_total_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # 提取特征（不使用adapter）
                with torch.no_grad():
                    features = self._extract_backbone_features(inputs)['features']  # [B, 768]

                # 前向传播量子识别器
                quantum_output = self._network.backbone.quantum_identifier(
                    features, task_id=self._cur_task, mode='training'
                )
                # quantum_output: (task_logits, readout, pqk_weights)
                task_logits = quantum_output[0]  # [B, max_tasks]
                readout = quantum_output[1]  # [B, 3]

                # 计算任务标签
                task_labels = torch.tensor([self.get_task_id_from_class(t.item()) for t in targets],
                                          device=self._device, dtype=torch.long)

                # 任务分类损失（交叉熵）
                loss_task = F.cross_entropy(task_logits[:, :self._cur_task+1], task_labels)

                # 总损失（只包含CE loss）
                total_loss = loss_task

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(quantum_params, max_norm=1.0)
                optimizer.step()

                # 统计
                epoch_total_loss += total_loss.item()

                # 计算精度
                with torch.no_grad():
                    preds = task_logits[:, :self._cur_task+1].argmax(dim=-1)
                    epoch_correct += (preds == task_labels).sum().item()
                    epoch_total += len(targets)

            scheduler.step()

            # 计算平均值
            avg_total_loss = epoch_total_loss / len(train_loader)
            accuracy = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0

            # 输出日志
            info = f"[INFO] Quantum Epoch {epoch + 1}/{self.quantum_epochs} => Total Loss {avg_total_loss:.4f}, Task CE {avg_total_loss:.4f}, Acc {accuracy:.2f}%"
            prog_bar.set_description(info)

        logging.info(info)

        # 验证阶段
        self._validate_quantum_identifier(train_loader)

    def _validate_quantum_identifier(self, loader):
        """验证量子识别器性能 - 使用task_head进行routing"""
        import time

        self._network.eval()
        correct = 0
        total = 0

        # 时间测量
        inference_times = []

        print(f"[DEBUG] using task_head for routing. Starting validation...")

        with torch.no_grad():
            for batch_idx, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # 提取特征
                features = self._extract_backbone_features(inputs)['features']

                # 测量推理时间
                synchronize(self._device)
                start_time = time.perf_counter()

                # 前向传播（inference模式）
                preds, confidences = self._network.backbone.quantum_identifier(
                    features, task_id=self._cur_task, mode='inference'
                )

                synchronize(self._device)
                end_time = time.perf_counter()

                # 记录时间（毫秒）
                inference_times.append((end_time - start_time) * 1000)

                # 计算任务标签
                task_labels = torch.tensor([self.get_task_id_from_class(t.item()) for t in targets],
                                          device=self._device, dtype=torch.long)

                # 计算准确率
                correct += (preds == task_labels).sum().item()
                total += len(targets)

        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # 统计推理时间
        import numpy as np
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        total_time = np.sum(inference_times)

        print(f"[Hard Routing] Quantum Task ID Accuracy: {accuracy:.2f}%")
        print(f"[Timing] Quantum Inference - Avg: {avg_time:.4f}ms, Std: {std_time:.4f}ms, Total: {total_time:.2f}ms")
        logging.info(f"[EVAL] routing accuracy={accuracy:.4f}%")
        logging.info(f"[TIMING] quantum inference avg={avg_time:.4f}ms, std={std_time:.4f}ms")

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        epochs = self.args["epochs"]
        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()

            losses = 0.0
            cls_losses = 0.0
            kd_losses = 0.0
            entropy_losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # 🎯 使用蒸馏模式的前向传播
                if self.use_distillation and self._cur_task > 0:
                    outputs = self._network.backbone.forward_train_with_distillation(inputs)

                    # 当前adapter的logits
                    logits = self._network.backbone.head(outputs['x'])[:, :self._total_classes]
                    logits[:, :self._known_classes] = float('-inf')

                    # 分类损失
                    loss_cls = F.cross_entropy(logits, targets.long())

                    # 🎯 知识蒸馏损失（使用PQK权重）
                    if outputs['old_logits'] is not None and outputs['final_weights'] is not None:
                        old_logits = outputs['old_logits']  # [B, num_old_adapters, num_classes]
                        pqk_weights = outputs['final_weights']  # [B, num_old_adapters] (PQK权重)

                        # 当前adapter对所有类的logits（用于蒸馏）
                        cur_logits_full = self._network.backbone.head(outputs['x'])  # [B, num_classes]

                        # 计算加权蒸馏损失
                        T = float(self.kd_temperature)
                        B, num_old_adapters, num_classes = old_logits.shape

                        # 计算 teacher probs: [B, K, C]
                        teacher_probs = F.softmax(old_logits / T, dim=-1)  # [B, num_old_adapters, num_classes]

                        # 计算 student log probs: expand to [B, K, C]
                        student_logits = cur_logits_full.unsqueeze(1).expand(-1, num_old_adapters, -1)  # [B, num_old_adapters, num_classes]
                        student_log_probs = F.log_softmax(student_logits / T, dim=-1)  # [B, num_old_adapters, num_classes]

                        # KL散度 per teacher per sample
                        kl_per_teacher = (teacher_probs * (torch.log(teacher_probs + 1e-12) - student_log_probs)).sum(dim=-1)  # [B, num_old_adapters]

                        # ============ PQK DEBUG PRINTS ============
                        print(f"[DEBUG] PQK weights for distillation (first sample): {pqk_weights[0].detach().cpu().numpy()}")
                        print(f"[DEBUG] KL per teacher (first sample): {kl_per_teacher[0].detach().cpu().numpy()}")
                        # ==========================================

                        # 使用PQK权重加权并平均
                        loss_kd = (pqk_weights * kl_per_teacher).sum(dim=1).mean() * (T ** 2)

                        # ============ PQK DEBUG PRINTS ============
                        print(f"[DEBUG] KD loss (weighted): {loss_kd.item():.6f}")
                        # ==========================================
                    else:
                        loss_kd = torch.tensor(0.0).to(self._device)

                    # 🎯 熵稀疏正则（移除，因为不再单独维护 gate_weights）
                    loss_entropy = torch.tensor(0.0).to(self._device)

                    # 总损失
                    loss = loss_cls + self.lambda_kd * loss_kd + loss_entropy

                    # 记录各部分损失
                    cls_losses += loss_cls.item()
                    kd_losses += loss_kd.item()
                    entropy_losses += loss_entropy.item()

                else:
                    # 第一个任务：标准训练（无蒸馏）
                    outputs = self._network(inputs, test=False)
                    logits = outputs["logits"][:, :self._total_classes]
                    logits[:, :self._known_classes] = float('-inf')
                    loss = F.cross_entropy(logits, targets.long())
                    cls_losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if self.use_distillation and self._cur_task > 0:
                info = "[INFO] Quantum Epoch {}/{} => Total Loss {:.4f}, CE {:.4f}, KD {:.4f}, Train_accy {:.2f}".format(
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        cls_losses / len(train_loader),
                        kd_losses / len(train_loader),
                        train_acc,
                    )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        epochs,
                        losses / len(train_loader),
                        train_acc,
                    )

            prog_bar.set_description(info)

        logging.info(info)

    def _extract_backbone_features(self, x):
        """提取backbone特征（使用第一个adapter）"""
        # 简化版本：直接通过当前backbone
        # 在实际实现中，这里应该调用backbone的无adapter版本
        return self._network.backbone.forward(x, test=False, quan=True)

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
            quantum_task_correct = 0  # 量子电路任务识别准确率
        self._network.eval()
        y_pred, y_true = [], []

        # 获取软路由配置
        use_soft_routing = self.args.get("use_soft_routing", False)
        temperature = self.args.get("soft_routing_temperature", 1.0)

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                outputs = self._network(inputs, test=True, use_soft_routing=use_soft_routing, temperature=temperature)
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

                    # 🎯 计算量子电路任务识别准确率（基于 task_head 预测）
                    if 'predicted_tasks' in outputs:
                        predicted_quantum_tasks = outputs['predicted_tasks']
                        # 真实任务ID：第0个任务是init_cls，后续任务每个inc个类
                        true_task_ids = torch.zeros_like(targets)
                        for i, target in enumerate(targets):
                            true_task_ids[i] = self.get_task_id_from_class(target.item())

                        quantum_task_correct += (predicted_quantum_tasks.cpu() == true_task_ids.cpu()).sum()

                    # calculate the accuracy of task_id (based on final classification)
                    pred_task_ids = (torch.max(logits, dim=1)[1] - self.init_cls) // self.inc + 1
                    task_correct += (pred_task_ids.cpu() == task_ids).sum()

                    pred_task_y = torch.max(task_logits, dim=1)[1]
                    task_acc += (pred_task_y.cpu() == targets).sum()
                    total += len(targets)



        if calc_task_acc:
            routing_type = "Soft Routing" if use_soft_routing else "Hard Routing"
            logging.info(f"[{routing_type}] Quantum Task ID Accuracy: {tensor2numpy(quantum_task_correct) * 100 / total:.2f}%")
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
                idx_dataset,
                batch_size=self.batch_size*3,
                shuffle=False,
                **loader_kwargs(self.args, self._device),
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
                self.cls_mean[class_idx] = torch.randn(feature_dim).to(self._ca_device) * 0.01
                if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                    self.cls_cov[class_idx] = torch.eye(feature_dim).to(self._ca_device)
                continue

            vectors = torch.cat(vectors, dim=0).to(self._ca_device)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                mean = features_per_cls.mean(dim=0)

                # 检查均值是否有效
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    logging.error(f"[_compute_mean] Class {class_idx}: computed mean contains NaN/Inf")
                    mean = torch.zeros_like(mean)

                self.cls_mean[class_idx] = mean.to(self._ca_device)

                # 计算协方差并添加正则化
                try:
                    cov = torch.cov(features_per_cls.T) + (torch.eye(mean.shape[-1]) * 1e-3).to(self._ca_device)

                    # 检查协方差是否有效
                    if torch.isnan(cov).any() or torch.isinf(cov).any():
                        logging.error(f"[_compute_mean] Class {class_idx}: computed cov contains NaN/Inf, using identity")
                        cov = torch.eye(mean.shape[-1]).to(self._ca_device)

                    self.cls_cov[class_idx] = cov
                except Exception as e:
                    logging.error(f"[_compute_mean] Class {class_idx}: cov computation failed: {e}, using identity")
                    self.cls_cov[class_idx] = torch.eye(mean.shape[-1]).to(self._ca_device)

            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                mean = features_per_cls.mean(dim=0)

                # 检查均值是否有效
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    logging.error(f"[_compute_mean] Class {class_idx}: computed mean contains NaN/Inf")
                    mean = torch.zeros_like(mean)

                self.cls_mean[class_idx] = mean.to(self._ca_device)

                # 计算方差并添加正则化
                try:
                    cov = torch.cov(features_per_cls.T) + (torch.eye(mean.shape[-1]) * 1e-3).to(self._ca_device)
                    var = torch.diag(cov)

                    # 检查方差是否有效
                    if torch.isnan(var).any() or torch.isinf(var).any():
                        logging.error(f"[_compute_mean] Class {class_idx}: computed var contains NaN/Inf, using ones")
                        var = torch.ones(mean.shape[-1]).to(self._ca_device)

                    self.cls_cov[class_idx] = var
                except Exception as e:
                    logging.error(f"[_compute_mean] Class {class_idx}: var computation failed: {e}, using ones")
                    self.cls_cov[class_idx] = torch.ones(mean.shape[-1]).to(self._ca_device)

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

                        cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._ca_device)
                        cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(self._ca_device)

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
                        cluster_means = [torch.tensor(np.mean(features_per_cls, axis=0), dtype=torch.float64).to(self._ca_device)]
                        cluster_vars = [torch.tensor(np.var(features_per_cls, axis=0), dtype=torch.float64).to(self._ca_device) + 1e-6]

                    self.cls_mean[class_idx] = cluster_means
                    self.cls_cov[class_idx] = cluster_vars

                except Exception as e:
                    logging.error(f"[_compute_mean] Class {class_idx}: clustering failed: {e}, using fallback")
                    # Fallback: 使用整体均值和方差
                    cluster_means = [torch.tensor(np.mean(features_per_cls, axis=0), dtype=torch.float64).to(self._ca_device)]
                    cluster_vars = [torch.tensor(np.var(features_per_cls, axis=0), dtype=torch.float64).to(self._ca_device) + 1e-6]
                    self.cls_mean[class_idx] = cluster_means
                    self.cls_cov[class_idx] = cluster_vars


    def classifer_align(self, model):
        model.train()

        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._ca_device)
                    cov = self.cls_cov[class_idx].to(self._ca_device)

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
                    cov = cov + torch.eye(cov.shape[0]).to(self._ca_device) * 1e-3

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

                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]

                loss = F.cross_entropy(logits, tgt)

                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"[CA] Epoch {epoch} batch {_iter}: loss is NaN/Inf, skipping")
                    continue

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()

                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()
                losses += loss.item()
                num_valid_batches += 1

            scheduler.step()

            # 避免除以0
            if num_valid_batches > 0 and total > 0:
                ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
                avg_loss = losses / num_valid_batches
            else:
                ca_acc = 0.0
                avg_loss = 0.0

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                avg_loss,
                ca_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
