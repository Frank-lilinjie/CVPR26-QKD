import json
import argparse
from trainer import train


def main():
    cli_args = setup_parser().parse_args()
    param = load_json(cli_args.config)

    args = dict(param)
    args["config"] = cli_args.config
    args["exp"] = cli_args.exp

    overrides = {
        "backend": cli_args.backend,
        "num_workers": cli_args.num_workers,
        "quantum_device": cli_args.quantum_device,
        "epochs": cli_args.epochs,
        "quantum_epochs": cli_args.quantum_epochs,
        "crct_epochs": cli_args.crct_epochs,
        "max_tasks": cli_args.max_tasks,
        "max_samples_per_class": cli_args.max_samples_per_class,
        "checkpoint_path": cli_args.checkpoint_path,
        "checkpoint_dir": cli_args.checkpoint_dir,
        "data_root": cli_args.data_root,
        "n_qubits": cli_args.n_qubits,
        "svd_dim": cli_args.svd_dim,
        "batch_size": cli_args.batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            args[key] = value

    if cli_args.device is not None:
        args["device"] = [cli_args.device]
    if cli_args.seed is not None:
        args["seed"] = [cli_args.seed]

    train(args)


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce quantum adapter incremental learning experiments.')
    parser.add_argument('--config', type=str, default='./exps/inr/quan_adapter.json',
                        help='Json file of settings.')
    parser.add_argument('--exp', type=str, default='npu_debug',
                        help='exp.')
    parser.add_argument('--device', type=str, default=None,
                        help='Single device, e.g. npu:0, cuda:0, cpu or auto.')
    parser.add_argument('--backend', choices=['auto', 'npu', 'cuda', 'cpu'], default=None,
                        help='Accelerator backend. Defaults to QKD_BACKEND or auto.')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers. Use 0 for the first NPU smoke test.')
    parser.add_argument('--quantum-device', default=None,
                        help='Device for the complex-valued quantum simulator: auto, cpu, npu:0 or same.')
    parser.add_argument('--n_qubits', type=int, default=None,
                        help='Override n_qubits in config.')
    parser.add_argument('--svd_dim', type=int, default=None,
                        help='Override svd_dim in config.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override adapter training epochs.')
    parser.add_argument('--quantum-epochs', type=int, default=None,
                        help='Override quantum task identifier epochs.')
    parser.add_argument('--crct-epochs', type=int, default=None,
                        help='Override classifier alignment epochs.')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Limit the number of incremental tasks (smoke testing).')
    parser.add_argument('--max-samples-per-class', type=int, default=None,
                        help='Limit train/test samples per class (smoke testing).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the config seed list with one seed.')
    parser.add_argument('--checkpoint-path', default=None,
                        help='Path to a ViT .npz checkpoint, or a directory containing it.')
    parser.add_argument('--checkpoint-dir', default=None,
                        help='Directory containing the B_16 ViT .npz checkpoints.')
    parser.add_argument('--data-root', default=None,
                        help='Common dataset root. Defaults to QKD_DATA_ROOT.')
    return parser


if __name__ == '__main__':
    main()

# 用量子电路改成任务识别器。
