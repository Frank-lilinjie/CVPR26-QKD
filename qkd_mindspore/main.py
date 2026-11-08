"""Command-line entry point for the MindSpore/Ascend QKD implementation.

Only configuration helpers are imported at module import time.  The trainer is
loaded lazily for ``python -m qkd_mindspore.main --help`` and ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import apply_cli_overrides, load_json, normalize_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "exps" / "smoke" / "mindspore_synthetic.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the QKD/QuanAdapter incremental-learning pipeline with "
            "MindSpore on Ascend 910B (CANN)."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="JSON experiment configuration (default: synthetic smoke config).",
    )
    parser.add_argument("--exp", default=None, help="Experiment name override.")
    parser.add_argument(
        "--framework",
        choices=("mindspore", "ms", "ascend"),
        default=None,
        help="Framework selector; this entry point always runs MindSpore.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, for example npu:0, Ascend:0, cpu, or auto.",
    )
    parser.add_argument(
        "--device-target",
        choices=("auto", "ascend", "npu", "cann", "cpu"),
        default=None,
        help="MindSpore device target override.",
    )
    parser.add_argument(
        "--ms-mode",
        "--mode",
        dest="ms_mode",
        choices=("pynative", "graph"),
        default=None,
        help="MindSpore execution mode.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name override.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root override. Defaults to QKD_DATA_ROOT or ./data.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="ViT .npz or MindSpore .ckpt checkpoint, or a directory containing it.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing the timm ViT .npz checkpoint.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Require and load a pretrained ViT checkpoint.",
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Allow random initialisation when no checkpoint is found.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Adapter epochs.")
    parser.add_argument(
        "--quantum-epochs", type=int, default=None, help="Quantum-router epochs."
    )
    parser.add_argument(
        "--crct-epochs", type=int, default=None, help="Classifier-alignment epochs."
    )
    parser.add_argument("--init-cls", type=int, default=None, help="Initial classes.")
    parser.add_argument(
        "--increment", type=int, default=None, help="Classes added per incremental task."
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Limit the number of tasks."
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Limit train/test samples per class for smoke testing.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Host prefetch workers.",
    )
    parser.add_argument("--n-qubits", type=int, default=None, help="Qubit count.")
    parser.add_argument(
        "--svd-dim", type=int, default=None, help="SVD feature dimension."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for metrics and checkpoints.",
    )
    parser.add_argument(
        "--log-dir", default=None, help="Root directory for log files."
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Disable per-task checkpoint and prototype files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the effective configuration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_config = load_json(args.config)
    apply_cli_overrides(raw_config, args)
    if args.framework is not None:
        raw_config["framework"] = args.framework
    if args.no_save_checkpoint:
        raw_config["save_checkpoint"] = False

    config = normalize_config(raw_config)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True, default=str))
        return 0

    # Keep MindSpore out of the import path until an actual run is requested.
    from .trainer import train

    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
