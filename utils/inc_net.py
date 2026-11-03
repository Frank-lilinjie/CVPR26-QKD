import copy
import logging
import torch
from torch import nn
import timm

checkpoint_path_in1k = "/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
checkpoint_path_in21k = "/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if '_quan_adapter' in name:
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
                quantum_layers=args.get("quantum_layers", 3),
                max_tasks=args.get("nb_tasks", 10),
                anchors_per_task=args.get("anchors_per_task", 3),
                ema_decay=args.get("ema_decay", 0.999),
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_quan_adapter":
                model = vit_quan_adapter.vit_base_patch16_224_quan_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_quan_adapter":
                model = vit_quan_adapter.vit_base_patch16_224_in21k_quan_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
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

    def forward(self, x, test=False, fc_only=False):
        res = self.backbone(x, test, fc_only)
        return res