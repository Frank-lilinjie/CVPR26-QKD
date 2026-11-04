import copy
import logging
import math
import torch
from torch import nn
import timm
from utils.paths import resolve_checkpoint_path


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    checkpoint_in1k = resolve_checkpoint_path("in1k", args)
    checkpoint_in21k = resolve_checkpoint_path("in21k", args)
    if '_prototype_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "prototype_adapter":
            from backbone import vit_prototype_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # Prototype related
                similarity_metric=args.get("similarity_metric", "cosine"),
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_in21k_prototype_adapter":
                model = vit_prototype_adapter.vit_base_patch16_224_in21k_prototype_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model
    elif '_linear_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "linear_adapter":
            from backbone import vit_linear_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # Linear Task Identifier related
                max_tasks=args.get("nb_tasks", 10),
                linear_hidden_dim=args.get("linear_hidden_dim", 256),
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_in21k_linear_adapter":
                model = vit_linear_adapter.vit_base_patch16_224_in21k_linear_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model
    elif '_quan_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "quan_adapter":
            from backbone import vit_quan_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                # Quantum related
                n_qubits=args.get("n_qubits", 6),
                svd_dim=args.get("svd_dim", 12),
                data_reuploading=args.get(
                    "data_reuploading",
                    max(1, math.ceil(args.get("svd_dim", 12) / max(1, args.get("n_qubits", 6))))
                ),
                quantum_layers=args.get("quantum_layers", 3),
                max_tasks=args.get("nb_tasks", 10),
                anchors_per_task=args.get("anchors_per_task", 3),
                ema_decay=args.get("ema_decay", 0.999),
                quantum_device=args.get("quantum_device", "auto"),
                svd_random_state=args.get("svd_random_state", 1993),
                _device=args["device"][0],
            )
            if name == "vit_base_patch16_224_quan_adapter":
                model = vit_quan_adapter.vit_base_patch16_224_quan_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config,
                    checkpoint_path=checkpoint_in1k)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_quan_adapter":
                model = vit_quan_adapter.vit_base_patch16_224_in21k_quan_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config,
                    checkpoint_path=checkpoint_in21k)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model
    elif '_quan_prompt' in name:
        if args["model_name"] == "quan_prompt":
            from backbone import vit_quan_prompt
            model = timm.create_model(
                    args["backbone_type"],
                    pretrained=args["pretrained"],
                    num_classes=args["nb_classes"],
                    drop_rate=args["drop"],
                    drop_path_rate=args["drop_path"],
                    drop_block_rate=None,
                    prompt_length=args["prompt_length"],
                    head_type=args["head_type"],
                    prompt_layer_idx=args["prompt_layer_idx"],
                    use_prefix_tune_for_prompt=args["use_prefix_tune_for_prompt"],
                    n_qubits=args["n_qubits"],
                    quantum_layers=args["quantum_layers"],
                    max_tasks=args["nb_tasks"],
                    anchors_per_task=args.get("anchors_per_task", 3),
                    ema_decay=args["ema_decay"],
                    checkpoint_path=checkpoint_in21k
            )
            return model
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class QuanAdapterNet(nn.Module):
    """Quantum Adapter Network - 集成量子任务识别和adapter的网络"""
    def __init__(self, args, pretrained):
        super(QuanAdapterNet,self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.out_dim = self.backbone.out_dim

    @property
    def feature_dim(self):
        return self.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False, fc_only=False, use_soft_routing=False, temperature=1.0):
        res = self.backbone(x, test, fc_only, use_soft_routing=use_soft_routing, temperature=temperature)
        return res


class LinearAdapterNet(nn.Module):
    """Linear Adapter Network - 集成线性任务识别和adapter的网络（消融实验baseline）"""
    def __init__(self, args, pretrained):
        super(LinearAdapterNet,self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.out_dim = self.backbone.out_dim

    @property
    def feature_dim(self):
        return self.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False, fc_only=False):
        res = self.backbone(x, test, fc_only)
        return res


class PrototypeAdapterNet(nn.Module):
    """Prototype Adapter Network - 集成原型任务识别和adapter的网络（消融实验baseline）"""
    def __init__(self, args, pretrained):
        super(PrototypeAdapterNet,self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.out_dim = self.backbone.out_dim

    @property
    def feature_dim(self):
        return self.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False, fc_only=False):
        res = self.backbone(x, test, fc_only)
        return res


class QuanPromptNet(nn.Module):
    """Quantum Prompt Network - 集成量子任务识别和prompt的网络"""
    def __init__(self, args, pretrained):
        super(QuanPromptNet,self).__init__()
        self.backbone = get_backbone(args, pretrained)

    def freeze(self):
        """冻结backbone但保持prompt和量子电路可训练"""
        self.backbone.freeze()

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False, fc_only=False):
        res = self.backbone(x, test, fc_only)
        return res
