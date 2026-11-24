"""Скрипт розподіленого та однопроцесного навчання / бенчмаркінгу PyTorch (ЛР6, Варіант 24)."""
import argparse
import datetime
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model_data import (
    B_GLOBAL,
    DATASETS,
    DS_KEY,
    LR0,
    MOMENTUM,
    N_EP,
    N_TRAIN,
    WIDTH_KEY,
    WIDTHS,
    Net,
    evaluate_model,
    load_split,
    n_params_formula,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", f"{DATASETS[DS_KEY][0]}.npz")


def safe_round(val, ndigits: int):
    if val is None:
        return None
    f = float(val)
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def build_model():
    _, K, C = DATASETS[DS_KEY]
    c1, c2, h = WIDTHS[WIDTH_KEY]
    torch.manual_seed(0)
    model = Net(C, c1, c2, h, K)
    n_p = sum(p.numel() for p in model.parameters())
    assert n_p == n_params_formula(C, c1, c2, h, K)
    return model, n_p


def init_dist_if_needed():
    if not dist.is_initialized():
        timeout = datetime.timedelta(seconds=30)
        restart_cnt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
        if restart_cnt > 0:
            rank = int(os.environ["RANK"])
            world = int(os.environ["WORLD_SIZE"])
            base_store = dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=int(os.environ["MASTER_PORT"]),
                world_size=world,
                is_master=False,
                timeout=timeout,
                wait_for_workers=False,
            )
            store = dist.PrefixStore(f"restart_{restart_cnt}/default_pg", base_store)
            dist.init_process_group(
                backend="gloo",
                store=store,
                rank=rank,
                world_size=world,
                timeout=timeout,
            )
        else:
            dist.init_process_group(backend="gloo", timeout=timeout)
    return dist.get_rank(), dist.get_world_size()


def next_batch_iter(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def mode_single_train(args):
    train_ds = load_split(DATA_PATH, "train", n=N_TRAIN)
    test_ds = load_split(DATA_PATH, "test")
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model, n_p = build_model()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()

    steps = 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            steps += 1

    acc, auc = evaluate_model(model, test_loader)
    out = {
        "mode": "single_train",
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "steps": steps,
        "n_params": n_p,
        "M_mib": round(4 * n_p / (1024 * 1024), 4),
        "acc": round(acc, 5),
        "auc": round(auc, 5),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[single_train] steps={steps} ACC={acc:.4f} AUC={auc:.4f}")


def mode_single_bench(args):
    train_ds = load_split(DATA_PATH, "train", n=N_TRAIN)
    g = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
    it = next_batch_iter(train_loader)

    model, _ = build_model()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for _ in range(10):
        xb, yb = next(it)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

    step_times = []
    t0 = time.perf_counter()
    for _ in range(50):
        s0 = time.perf_counter()
        xb, yb = next(it)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
        s1 = time.perf_counter()
        step_times.append(s1 - s0)
    t_total = time.perf_counter() - t0

    t_step_ms = statistics.median(step_times) * 1000.0
    x_rps = (50 * args.batch_size) / t_total
    out = {
        "mode": "single_bench",
        "batch_size": args.batch_size,
        "t_step_ms": round(t_step_ms, 3),
        "t_total_50_s": round(t_total, 4),
        "X": round(x_rps, 3),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[single_bench] B={args.batch_size} t_step={t_step_ms:.2f} ms X={x_rps:.2f} samp/s")


def mode_equiv(args):
    rank, world = init_dist_if_needed()
    train_ds = load_split(DATA_PATH, "train", n=N_TRAIN)

    # Перший батч із B = B_GLOBAL прикладів
    x_full, y_full = train_ds[:B_GLOBAL]
    x_loc = x_full[rank::world]
    y_loc = y_full[rank::world]

    model, _ = build_model()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    loss_fn = nn.CrossEntropyLoss()

    # Еталонний градієнт g на всьому батчі (обчислюється на rank 0 окремою копією з тими самими вагами)
    ref_grads = None
    if rank == 0:
        ref_model, _ = build_model()
        with torch.no_grad():
            for pr, p0 in zip(ref_model.parameters(), model.parameters()):
                pr.copy_(p0.data)
        ref_model.train()
        ref_model.zero_grad()
        loss_fn(ref_model(x_full), y_full).backward()
        ref_grads = [pr.grad.detach().clone() for pr in ref_model.parameters()]

    # Паралельний градієнт g_bar з ручним all-reduce
    model.train()
    model.zero_grad()
    loss_fn(model(x_loc), y_loc).backward()
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world

    max_rel_diff = 0.0
    if rank == 0:
        for g_ref, p_loc in zip(ref_grads, model.parameters()):
            g_par = p_loc.grad.detach()
            num = (g_ref - g_par).abs().max().item()
            den = g_ref.abs().max().item()
            if den > 0:
                max_rel_diff = max(max_rel_diff, num / den)

    # 50 кроків навчання з DistributedSampler та ручною синхронізацією
    b = B_GLOBAL // world
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(train_ds, batch_size=b, sampler=sampler, drop_last=True)
    it = next_batch_iter(loader, sampler=sampler)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)

    for _ in range(50):
        xb, yb = next(it)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world
        opt.step()

    # Збір сум усіх ваг у float64 через all_gather
    w_sum = sum(p.detach().double().sum().item() for p in model.parameters())
    t_local = torch.tensor([w_sum], dtype=torch.float64)
    gathered = [torch.zeros(1, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(gathered, t_local)

    if rank == 0:
        sums = [float(t.item()) for t in gathered]
        weight_sum_diff = max(sums) - min(sums)
        out = {
            "mode": "equiv",
            "world_size": world,
            "B": B_GLOBAL,
            "b": b,
            "max_grad_rel_diff": max_rel_diff,
            "weight_sums_50_steps": sums,
            "weight_sum_diff_50_steps": weight_sum_diff,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[equiv] max_grad_rel_diff={max_rel_diff:.3e}, weight_sum_diff={weight_sum_diff:.3e}")

    dist.barrier()
    dist.destroy_process_group()


def mode_manual_bench(args):
    rank, world = init_dist_if_needed()
    train_ds = load_split(DATA_PATH, "train", n=N_TRAIN)
    b = args.batch_size
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(train_ds, batch_size=b, sampler=sampler, drop_last=True)
    it = next_batch_iter(loader, sampler=sampler)

    model, _ = build_model()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for _ in range(10):
        xb, yb = next(it)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world
        opt.step()

    dist.barrier()
    step_times = []
    sync_times = []
    t0 = time.perf_counter()
    for _ in range(50):
        s0 = time.perf_counter()
        xb, yb = next(it)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        sy0 = time.perf_counter()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world
        sy1 = time.perf_counter()
        opt.step()
        s1 = time.perf_counter()
        step_times.append(s1 - s0)
        sync_times.append(sy1 - sy0)
    t_total = time.perf_counter() - t0

    if rank == 0:
        t_step_ms = statistics.median(step_times) * 1000.0
        t_sync_ms = statistics.median(sync_times) * 1000.0
        x_rps = (50 * world * b) / t_total
        out = {
            "mode": "manual_bench",
            "world_size": world,
            "b": b,
            "B": world * b,
            "t_step_ms": round(t_step_ms, 3),
            "t_sync_ms": round(t_sync_ms, 3),
            "comm_share": round(t_sync_ms / t_step_ms, 4),
            "t_total_50_s": round(t_total, 4),
            "X": round(x_rps, 3),
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[manual_bench] P={world} b={b} t_step={t_step_ms:.2f} ms t_sync={t_sync_ms:.2f} ms X={x_rps:.2f} samp/s")

    dist.barrier()
    dist.destroy_process_group()


def mode_ddp_bench(args):
    rank, world = init_dist_if_needed()
    train_ds = load_split(DATA_PATH, "train", n=N_TRAIN)
    b = args.batch_size
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(train_ds, batch_size=b, sampler=sampler, drop_last=True)
    it = next_batch_iter(loader, sampler=sampler)

    base_model, _ = build_model()
    model = DDP(base_model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for _ in range(10):
        xb, yb = next(it)
        opt.zero_grad()
        if args.no_sync:
            with model.no_sync():
                loss_fn(model(xb), yb).backward()
        else:
            loss_fn(model(xb), yb).backward()
        opt.step()

    dist.barrier()
    step_times = []
    t0 = time.perf_counter()
    for _ in range(50):
        s0 = time.perf_counter()
        xb, yb = next(it)
        opt.zero_grad()
        if args.no_sync:
            with model.no_sync():
                loss_fn(model(xb), yb).backward()
        else:
            loss_fn(model(xb), yb).backward()
        opt.step()
        s1 = time.perf_counter()
        step_times.append(s1 - s0)
    t_total = time.perf_counter() - t0

    if rank == 0:
        t_step_ms = statistics.median(step_times) * 1000.0
        x_rps = (50 * world * b) / t_total
        out = {
            "mode": "ddp_bench",
            "no_sync": args.no_sync,
            "world_size": world,
            "b": b,
            "B": world * b,
            "t_step_ms": round(t_step_ms, 3),
            "t_total_50_s": round(t_total, 4),
            "X": round(x_rps, 3),
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[ddp_bench no_sync={args.no_sync}] P={world} b={b} t_step={t_step_ms:.2f} ms X={x_rps:.2f} samp/s")

    dist.barrier()
    dist.destroy_process_group()


def mode_ddp_train(args):
    rank, world = init_dist_if_needed()
    train_ds = load_split(DATA_PATH, "train", n=N_TRAIN)
    test_ds = load_split(DATA_PATH, "test")
    b = args.batch_size
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    train_loader = DataLoader(train_ds, batch_size=b, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    base_model, _ = build_model()

    start_epoch = 0
    skip_steps = 0
    step = 0
    history_losses = {}
    ckpt_time_map = {}
    resumed_from_ckpt = False
    first_step_after_resume_done = None
    ckpt_state = None

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt_state = torch.load(args.ckpt_path, weights_only=True)
        base_model.load_state_dict(ckpt_state["model"])
        resumed_from_ckpt = True

    model = DDP(base_model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()

    if resumed_from_ckpt and ckpt_state is not None:
        opt.load_state_dict(ckpt_state["opt"])
        start_epoch = int(ckpt_state["epoch"])
        skip_steps = int(ckpt_state["step_in_epoch"])
        step = int(ckpt_state["step"])
        if os.path.exists(args.out):
            try:
                prev = json.load(open(args.out, encoding="utf-8"))
                history_losses = {str(k): v for k, v in prev.get("losses_by_step", {}).items()}
                ckpt_time_map = prev.get("ckpt_times", {})
            except Exception:
                pass
        if rank == 0:
            print(f"[ddp_train] Resumed from {args.ckpt_path}: epoch={start_epoch}, step_in_epoch={skip_steps}, step={step}", flush=True)

    last_epoch_losses = []

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        steps_in_this_epoch = len(train_loader)
        for i, (xb, yb) in enumerate(train_loader):
            if epoch == start_epoch and i < skip_steps:
                continue

            next_step = step + 1
            # Імітація відмови на початку кроку s_f на процесі з rank = P_max - 1
            if args.fail_step > 0 and next_step == args.fail_step and rank == (world - 1):
                if args.fail_marker and not os.path.exists(args.fail_marker):
                    with open(args.fail_marker, "w", encoding="utf-8") as mf:
                        json.dump({"t_f": time.time(), "step": next_step, "rank": rank}, mf)
                    print(f"[rank {rank}] Simulating crash os._exit(1) at start of step {next_step}", flush=True)
                    os._exit(1)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            step += 1

            # Запис втрати на кроці на rank 0
            loss_t = loss.detach().clone()
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            avg_loss = float(loss_t.item() / world)

            if rank == 0:
                if resumed_from_ckpt and first_step_after_resume_done is None:
                    first_step_after_resume_done = time.time()
                history_losses[str(step)] = safe_round(avg_loss, 6)
                if epoch == args.epochs - 1:
                    last_epoch_losses.append(avg_loss)

            if args.ckpt_every > 0 and args.ckpt_path and (step % args.ckpt_every == 0):
                if rank == 0:
                    save_ep = epoch
                    save_i = i + 1
                    if save_i >= steps_in_this_epoch:
                        save_ep += 1
                        save_i = 0
                    state = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "epoch": save_ep,
                        "step_in_epoch": save_i,
                        "step": step,
                    }
                    tmp_p = args.ckpt_path + ".tmp"
                    torch.save(state, tmp_p)
                    os.replace(tmp_p, args.ckpt_path)
                    ckpt_time_map[str(step)] = time.time()
                    if args.out and (not resumed_from_ckpt) and args.fail_step > 0:
                        with open(args.out, "w", encoding="utf-8") as f:
                            json.dump({"losses_by_step": history_losses, "ckpt_times": ckpt_time_map}, f)
                dist.barrier()

            if rank == 0 and args.out and (not resumed_from_ckpt) and args.fail_step > 0 and step % 5 == 0:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump({"losses_by_step": history_losses, "ckpt_times": ckpt_time_map}, f)

        skip_steps = 0

    if rank == 0:
        acc, auc = evaluate_model(model.module, test_loader)
        finite_last = [v for v in last_epoch_losses if math.isfinite(v)]
        if last_epoch_losses and len(finite_last) < len(last_epoch_losses):
            max_loss_last_epoch = float("nan")
        elif finite_last:
            max_loss_last_epoch = max(finite_last)
        else:
            fallback_vals = [
                v for k, v in history_losses.items()
                if v is not None and int(k) > (args.epochs - 1) * (step // args.epochs)
            ]
            max_loss_last_epoch = max(fallback_vals) if fallback_vals else None
        out = {
            "mode": "ddp_train",
            "world_size": world,
            "b": b,
            "B": world * b,
            "lr": args.lr,
            "epochs": args.epochs,
            "total_steps": step,
            "acc": safe_round(acc, 5),
            "auc": safe_round(auc, 5),
            "max_loss_last_epoch": safe_round(max_loss_last_epoch, 6),
            "resumed_from_ckpt": resumed_from_ckpt,
            "ckpt_times": ckpt_time_map,
            "first_step_after_resume_unix": first_step_after_resume_done,
            "losses_by_step": history_losses,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        acc_s = f"{acc:.4f}" if math.isfinite(acc) else "NaN"
        auc_s = f"{auc:.4f}" if math.isfinite(auc) else "NaN"
        maxl_s = f"{max_loss_last_epoch:.4f}" if (max_loss_last_epoch is not None and math.isfinite(max_loss_last_epoch)) else "NaN"
        print(f"[ddp_train] P={world} B={world*b} lr={args.lr} steps={step} ACC={acc_s} AUC={auc_s} max_loss_last={maxl_s}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["single_train", "single_bench", "equiv", "manual_bench", "ddp_bench", "ddp_train"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=B_GLOBAL)
    ap.add_argument("--lr", type=float, default=LR0)
    ap.add_argument("--epochs", type=int, default=N_EP)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--ckpt-path", type=str, default="")
    ap.add_argument("--fail-step", type=int, default=0)
    ap.add_argument("--fail-marker", type=str, default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mode == "single_train":
        mode_single_train(args)
    elif args.mode == "single_bench":
        mode_single_bench(args)
    elif args.mode == "equiv":
        mode_equiv(args)
    elif args.mode == "manual_bench":
        mode_manual_bench(args)
    elif args.mode == "ddp_bench":
        mode_ddp_bench(args)
    elif args.mode == "ddp_train":
        mode_ddp_train(args)


if __name__ == "__main__":
    main()
