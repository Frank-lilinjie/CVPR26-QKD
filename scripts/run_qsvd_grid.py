#!/usr/bin/env python3
"""Lambda grid scheduler for CUDA, Ascend NPU and CPU."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque


ACC_PATTERN = re.compile(r"Average Accuracy \(CNN\)[^0-9+\-]*([+\-]?(?:\d+\.\d+|\d+))")


def _format_float_token(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


@dataclass
class TaskItem:
    lambda_kd: float
    lambda_sparse: float
    retries: int = 0

    @property
    def task_id(self) -> str:
        return f"kd{_format_float_token(self.lambda_kd)}_sp{_format_float_token(self.lambda_sparse)}"

    @property
    def exp_name(self) -> str:
        return f"HyperParameter/lambda_kd_sparse/{self.task_id}"


@dataclass
class RunningJob:
    task: TaskItem
    process: subprocess.Popen[Any]
    config_path: Path
    device_id: str
    started_at: int


def build_grid(lambda_kd_values: list[float], lambda_sparse_values: list[float]) -> list[TaskItem]:
    tasks: list[TaskItem] = []
    for lambda_kd in lambda_kd_values:
        for lambda_sparse in lambda_sparse_values:
            tasks.append(TaskItem(lambda_kd=lambda_kd, lambda_sparse=lambda_sparse))
    return tasks


def query_gpu_memory() -> dict[int, dict[str, float]] | None:
    """Return NVIDIA memory information, or None when unavailable.

    The scheduler must remain usable on Ascend hosts where ``nvidia-smi`` is
    absent. In that case it falls back to concurrency-slot scheduling.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    result: dict[int, dict[str, float]] = {}
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        idx_raw, total_raw, used_raw, free_raw = [part.strip() for part in line.split(",")]
        idx = int(idx_raw)
        total_mib = float(total_raw)
        used_mib = float(used_raw)
        free_mib = float(free_raw)
        result[idx] = {
            "total_gb": total_mib / 1024.0,
            "used_gb": used_mib / 1024.0,
            "free_gb": free_mib / 1024.0,
        }
    return result


def make_temp_config(base_cfg: dict[str, Any], task: TaskItem, device_id: str, tmp_config_dir: Path) -> Path:
    cfg = dict(base_cfg)
    cfg["lambda_kd"] = task.lambda_kd
    cfg["lambda_sparse"] = task.lambda_sparse
    cfg["device"] = [device_id]

    tmp_config_dir.mkdir(parents=True, exist_ok=True)
    device_token = re.sub(r"[^A-Za-z0-9_-]+", "", device_id) or "device"
    config_path = tmp_config_dir / f"quan_adapter_{task.task_id}_{device_token}.json"
    config_path.write_text(json.dumps(cfg, indent=4, ensure_ascii=True) + "\n", encoding="utf-8")
    return config_path


def resolve_device_spec(device_id: str, backend: str) -> str:
    """Build an explicit device spec while preserving full ``npu:0`` forms."""
    device_id = str(device_id).strip()
    if ":" in device_id or device_id == "cpu":
        return device_id
    if backend == "auto":
        return device_id
    return f"{backend}:{device_id}"


def launch_job(
    repo_root: Path,
    config_path: Path,
    task: TaskItem,
    device_id: str,
    backend: str,
) -> subprocess.Popen[Any]:
    device_spec = resolve_device_spec(device_id, backend)
    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--config",
        str(config_path),
        "--exp",
        task.exp_name,
        "--device",
        device_spec,
    ]
    if backend != "auto":
        cmd.extend(["--backend", backend])
    print(f"[DISPATCH] task={task.task_id} cmd={' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=repo_root)


def extract_final_avg_acc(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = ACC_PATTERN.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def upsert_result_csv(csv_path: Path, lambda_kd: float, lambda_sparse: float, avg_acc: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    kd_str = f"{lambda_kd:.6f}"
    sparse_str = f"{lambda_sparse:.6f}"
    key = (kd_str, sparse_str)
    rows: list[dict[str, str]] = []

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and {"lambda_kd", "lambda_sparse", "Average Accuracy"}.issubset(reader.fieldnames):
                for row in reader:
                    rows.append(
                        {
                            "lambda_kd": str(row.get("lambda_kd", "")).strip(),
                            "lambda_sparse": str(row.get("lambda_sparse", "")).strip(),
                            "Average Accuracy": str(row.get("Average Accuracy", "")).strip(),
                        }
                    )
            else:
                f.seek(0)
                plain_reader = csv.reader(f)
                for row in plain_reader:
                    if len(row) < 3:
                        continue
                    rows.append(
                        {
                            "lambda_kd": row[0].strip(),
                            "lambda_sparse": row[1].strip(),
                            "Average Accuracy": row[2].strip(),
                        }
                    )

    updated = False
    for row in rows:
        if (row["lambda_kd"], row["lambda_sparse"]) == key:
            row["Average Accuracy"] = f"{avg_acc:.6f}"
            updated = True
            break

    if not updated:
        rows.append(
            {
                "lambda_kd": kd_str,
                "lambda_sparse": sparse_str,
                "Average Accuracy": f"{avg_acc:.6f}",
            }
        )

    rows.sort(key=lambda x: (float(x["lambda_kd"]), float(x["lambda_sparse"])))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda_kd", "lambda_sparse", "Average Accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def read_completed_tasks(csv_path: Path) -> set[str]:
    done: set[str] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"lambda_kd", "lambda_sparse"}.issubset(reader.fieldnames):
            for row in reader:
                kd_str = str(row.get("lambda_kd", "")).strip()
                sparse_str = str(row.get("lambda_sparse", "")).strip()
                if kd_str and sparse_str:
                    task = TaskItem(lambda_kd=float(kd_str), lambda_sparse=float(sparse_str))
                    done.add(task.task_id)
        else:
            f.seek(0)
            plain_reader = csv.reader(f)
            for row in plain_reader:
                if len(row) < 2:
                    continue
                task = TaskItem(lambda_kd=float(row[0].strip()), lambda_sparse=float(row[1].strip()))
                done.add(task.task_id)
    return done


def find_task_log(logs_root: Path, model_name: str, dataset: str, task: TaskItem) -> Path | None:
    exp_path = Path("HyperParameter") / "lambda_kd_sparse" / task.task_id
    candidate_root = logs_root / model_name / exp_path / dataset
    if not candidate_root.exists():
        return None
    log_files = list(candidate_root.glob("**/*.log"))
    if not log_files:
        return None
    log_files.sort(key=lambda p: p.stat().st_mtime)
    return log_files[-1]


def write_run_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def pop_next_task(pending: Deque[TaskItem], running_task_ids: set[str]) -> TaskItem | None:
    if not pending:
        return None
    for _ in range(len(pending)):
        task = pending.popleft()
        if task.task_id in running_task_ids:
            pending.append(task)
            continue
        return task
    return None


def scheduler_loop(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    base_config_path = Path(args.base_config).resolve()
    results_csv = Path(args.result_csv).resolve()
    tmp_config_dir = Path(args.tmp_config_dir).resolve()
    logs_root = Path(args.logs_root).resolve()
    run_state_path = Path(args.run_state).resolve()

    base_cfg = json.loads(base_config_path.read_text(encoding="utf-8"))
    model_name = str(base_cfg.get("model_name", "quan_adapter"))
    dataset = str(base_cfg.get("dataset", "imageneta"))

    all_tasks = build_grid(args.lambda_kds, args.lambda_sparses)
    completed = read_completed_tasks(results_csv)
    pending: Deque[TaskItem] = deque(task for task in all_tasks if task.task_id not in completed)

    device_ids = [str(device_id) for device_id in args.device_ids]
    running: dict[str, list[RunningJob]] = {device_id: [] for device_id in device_ids}

    run_state: dict[str, Any] = {
        "pending": [task.task_id for task in pending],
        "running": {device_id: [] for device_id in device_ids},
        "finished": sorted(completed),
        "failed": {},
        "updated_at": int(time.time()),
    }
    write_run_state(run_state_path, run_state)

    print(f"[INIT] total={len(all_tasks)} completed={len(completed)} remaining={len(pending)}")

    while pending or any(running[device_id] for device_id in device_ids):
        for device_id in device_ids:
            survivors: list[RunningJob] = []
            for job in running[device_id]:
                ret = job.process.poll()
                if ret is None:
                    survivors.append(job)
                    continue

                task = job.task
                print(f"[FINISH] task={task.task_id} device={device_id} returncode={ret}")

                if ret != 0:
                    if task.retries < args.max_retries:
                        task.retries += 1
                        pending.appendleft(task)
                        print(f"[RETRY] task={task.task_id} retry={task.retries}/{args.max_retries}")
                    else:
                        run_state["failed"][task.task_id] = f"exit_code={ret}"
                        print(f"[FAIL] task={task.task_id} exceeded retries")
                    continue

                avg_acc: float | None = None
                log_path: Path | None = None
                for _ in range(args.log_retry_count):
                    log_path = find_task_log(logs_root, model_name, dataset, task)
                    if log_path is not None:
                        avg_acc = extract_final_avg_acc(log_path)
                        if avg_acc is not None:
                            break
                    time.sleep(args.log_retry_interval)

                if avg_acc is None:
                    msg = "missing_final_accuracy"
                    if log_path is not None:
                        msg += f" log={log_path}"
                    if task.retries < args.max_retries:
                        task.retries += 1
                        pending.appendleft(task)
                        print(f"[RETRY] task={task.task_id} reason={msg} retry={task.retries}/{args.max_retries}")
                    else:
                        run_state["failed"][task.task_id] = msg
                        print(f"[FAIL] task={task.task_id} reason={msg}")
                    continue

                upsert_result_csv(results_csv, task.lambda_kd, task.lambda_sparse, avg_acc)
                run_state["finished"] = sorted(set(run_state["finished"]) | {task.task_id})
                print(f"[RECORDED] task={task.task_id} avg_acc={avg_acc:.6f}")

            running[device_id] = survivors

        dispatched_any = False
        memory_info: dict[int, dict[str, float]] | None = None
        if args.backend == "cuda":
            memory_info = query_gpu_memory()
            if memory_info is None:
                print("[WARN] nvidia-smi unavailable; using concurrency-slot scheduling")

        while pending:
            candidates: list[tuple[str, float, int]] = []
            for device_id in device_ids:
                active_jobs = len(running[device_id])
                if active_jobs >= args.max_jobs_per_gpu:
                    continue
                free_gb = 0.0
                if memory_info is not None:
                    memory_index = int(device_id.split(":")[-1]) if device_id.split(":")[-1].isdigit() else 0
                    free_gb = memory_info.get(memory_index, {}).get("free_gb", 0.0)
                    if free_gb < args.free_mem_threshold_gb:
                        continue
                candidates.append((device_id, free_gb, active_jobs))

            target_device: str | None = None
            if candidates:
                candidates.sort(key=lambda item: (item[2], -item[1], str(item[0])))
                target_device = candidates[0][0]

            if target_device is None:
                break

            running_task_ids = {
                job.task.task_id
                for device_id in device_ids
                for job in running[device_id]
            }
            task = pop_next_task(pending, running_task_ids)
            if task is None:
                break
            cfg_path = make_temp_config(base_cfg, task, target_device, tmp_config_dir)
            proc = launch_job(repo_root, cfg_path, task, target_device, args.backend)
            running[target_device].append(
                RunningJob(
                    task=task,
                    process=proc,
                    config_path=cfg_path,
                    device_id=target_device,
                    started_at=int(time.time()),
                )
            )
            dispatched_any = True

            print(
                f"[RUNNING] device={target_device} jobs={len(running[target_device])}/{args.max_jobs_per_gpu} "
                f"pending={len(pending)}"
            )

            time.sleep(args.stabilization_seconds)

        run_state["pending"] = [task.task_id for task in pending]
        run_state["running"] = {
            device_id: [
                {
                    "task_id": job.task.task_id,
                    "pid": job.process.pid,
                    "config": str(job.config_path),
                    "started_at": job.started_at,
                    "retries": job.task.retries,
                }
                for job in running[device_id]
            ]
            for device_id in device_ids
        }
        run_state["updated_at"] = int(time.time())
        write_run_state(run_state_path, run_state)

        if pending and not dispatched_any:
            has_free_slot = any(
                len(running[device_id]) < args.max_jobs_per_gpu
                for device_id in device_ids
            )
            if memory_info is not None and has_free_slot:
                print(
                    f"[PAUSE] no device has >= {args.free_mem_threshold_gb:.1f}GB free, "
                    f"sleep {args.memory_pause_seconds}s"
                )
                time.sleep(args.memory_pause_seconds)
            else:
                time.sleep(args.poll_interval)
            continue

        if not pending and any(running[device_id] for device_id in device_ids):
            print("[DRAINING] waiting running jobs to finish")
            time.sleep(args.poll_interval)
            continue

        if pending:
            time.sleep(args.poll_interval)

    failed_count = len(run_state.get("failed", {}))
    print(f"[DONE] all tasks processed, failed={failed_count}")
    if failed_count:
        for task_id, reason in run_state["failed"].items():
            print(f"[FAILED_TASK] {task_id}: {reason}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run lambda_kd/lambda_sparse grids on CUDA, Ascend NPU or CPU.")
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--base-config", default=str(repo_root / "exps/inr/quan_adapter.json"))
    parser.add_argument("--result-csv", default=str(repo_root / "results/result.csv"))
    parser.add_argument("--tmp-config-dir", default=str(repo_root / "tmp_configs"))
    parser.add_argument("--logs-root", default=str(repo_root / "logs"))
    parser.add_argument("--run-state", default=str(repo_root / "results/run_state.json"))
    parser.add_argument("--backend", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    parser.add_argument("--device-ids", nargs="+", default=None,
                        help="Device indices or full specs, e.g. 0 1 or npu:0 npu:1.")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None,
                        help="Deprecated alias for --device-ids.")
    parser.add_argument("--lambda-kds", nargs="+", type=float, default=[0.5, 0.8, 1.0, 1.2, 1.5])
    parser.add_argument("--lambda-sparses", nargs="+", type=float, default=[0.005, 0.01, 0.05, 0.1, 0.2])
    parser.add_argument("--free-mem-threshold-gb", type=float, default=4.0)
    parser.add_argument("--max-jobs-per-device", "--max-jobs-per-gpu", dest="max_jobs_per_gpu",
                        type=int, default=1)
    parser.add_argument("--stabilization-seconds", type=int, default=20)
    parser.add_argument("--memory-pause-seconds", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--log-retry-count", type=int, default=5)
    parser.add_argument("--log-retry-interval", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    if args.device_ids is None:
        args.device_ids = [str(value) for value in args.gpu_ids] if args.gpu_ids else ["0"]
    if not args.device_ids:
        parser.error("--device-ids must contain at least one device")
    if args.max_jobs_per_gpu < 1:
        parser.error("--max-jobs-per-device must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    scheduler_loop(args)


if __name__ == "__main__":
    main()
