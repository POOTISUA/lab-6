"""Оркестратор експериментів ЛР6 (Варіант 24: Д6 OCTMNIST, Ш3, P_max=4, B=192, K_c=15, s_f=37)."""
import argparse
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def run_cmd(cmd: list):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["GLOO_SOCKET_IFNAME"] = "lo"
    print(">>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")


def torchrun(nproc: int, script_args: list, max_restarts: int = 0):
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--max-restarts={max_restarts}",
        "--rdzv-backend=static",
        "--master-addr=127.0.0.1",
        "--master-port=29500",
        "--node-rank=0",
        os.path.join(ROOT, "train_ddp.py"),
    ] + script_args
    run_cmd(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-step5", action="store_true", help="Виконати лише крок 5 (відновлення з контрольної точки)")
    args = ap.parse_args()

    import psutil
    import torch
    from download_data import ensure_data
    from model_data import B_GLOBAL, B_LOCAL, DATASETS, DS_KEY, K_C, LR0, N_EP, N_TRAIN, P_MAX, S_F, WIDTH_KEY, WIDTHS, n_params_formula

    ensure_data()

    _, K, C = DATASETS[DS_KEY]
    c1, c2, h = WIDTHS[WIDTH_KEY]
    n_p = n_params_formula(C, c1, c2, h, K)
    c_phys_detected = psutil.cpu_count(logical=False) or 8
    c_logical_detected = psutil.cpu_count(logical=True) or 12
    c_phys = int(os.environ.get("HOST_C_PHYS", c_phys_detected))
    c_logical = int(os.environ.get("HOST_C_LOGICAL", c_logical_detected))
    container_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    ram_gb = float(os.environ.get("HOST_RAM_GB", container_ram_gb))
    n_th = max(1, c_phys // P_MAX)
    os_str = (
        f"Linux {platform.release()} (Docker Python {sys.version.split()[0]} / WSL2, Windows 11)"
        if os.environ.get("HOST_C_PHYS")
        else platform.platform()
    )

    if not args.only_step5:
        env_info = {
            "variant": 24,
            "cpu": "12th Gen Intel(R) Core(TM) i5-12450H",
            "c_phys": c_phys,
            "c_logical": c_logical,
            "ram_gb": ram_gb,
            "container_ram_gb": container_ram_gb,
            "os": os_str,
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "n_th": n_th,
            "dataset": f"{DS_KEY} ({DATASETS[DS_KEY][0]})",
            "K": K,
            "C": C,
            "widths": f"{c1} / {c2} / {h}",
            "n_params": n_p,
            "M_mib": round(4 * n_p / (1024 ** 2), 4),
            "N": N_TRAIN,
            "B": B_GLOBAL,
            "P_max": P_MAX,
            "b": B_LOCAL,
            "K_c": K_C,
            "s_f": S_F,
        }
        json.dump(env_info, open(os.path.join(RES, "env.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # ---------- Крок 1: 1 процес без torch.distributed ----------
        run_cmd([sys.executable, "train_ddp.py", "--mode", "single_train", "--threads", str(n_th),
                 "--batch-size", str(B_GLOBAL), "--lr", str(LR0), "--epochs", str(N_EP),
                 "--out", os.path.join(RES, "step1_single_train.json")])
        run_cmd([sys.executable, "train_ddp.py", "--mode", "single_bench", "--threads", str(n_th),
                 "--batch-size", str(B_GLOBAL), "--lr", str(LR0),
                 "--out", os.path.join(RES, "step1_single_bench.json")])

        # ---------- Крок 2: еквівалентність, ручна синхронізація, навчання DDP на P_max = 4 ----------
        torchrun(P_MAX, ["--mode", "equiv", "--threads", str(n_th), "--lr", str(LR0),
                         "--out", os.path.join(RES, "step2_equiv.json")])
        torchrun(P_MAX, ["--mode", "manual_bench", "--threads", str(n_th), "--batch-size", str(B_LOCAL), "--lr", str(LR0),
                         "--out", os.path.join(RES, "step2_manual_bench.json")])
        torchrun(P_MAX, ["--mode", "ddp_train", "--threads", str(n_th), "--batch-size", str(B_LOCAL), "--lr", str(LR0),
                         "--epochs", str(N_EP), "--out", os.path.join(RES, "step2_ddp_train.json")])

        # ---------- Крок 3: Strong scaling (B = 192 const, P = 1..4 через torchrun з DDP; P=P_max також для Таблиці 2) ----------
        for p in range(1, P_MAX + 1):
            b_p = B_GLOBAL // p
            torchrun(p, ["--mode", "ddp_bench", "--threads", str(n_th), "--batch-size", str(b_p), "--lr", str(LR0),
                         "--out", os.path.join(RES, f"step3_strong_p{p}.json")])
            torchrun(p, ["--mode", "ddp_bench", "--no-sync", "--threads", str(n_th), "--batch-size", str(b_p), "--lr", str(LR0),
                         "--out", os.path.join(RES, f"step3_strong_nosync_p{p}.json")])

        # ---------- Крок 4: Weak scaling (b = 48 const, P = 1..4) та якість за різних B_P і lr ----------
        for p in range(1, P_MAX):
            torchrun(p, ["--mode", "ddp_bench", "--threads", str(n_th), "--batch-size", str(B_LOCAL), "--lr", str(LR0),
                         "--out", os.path.join(RES, f"step4_weak_p{p}.json")])
        # Якість P = 1, b = 48, lr = 0.02
        torchrun(1, ["--mode", "ddp_train", "--threads", str(n_th), "--batch-size", str(B_LOCAL), "--lr", str(LR0),
                     "--epochs", str(N_EP), "--out", os.path.join(RES, "step4_train_p1_b48.json")])
        # Якість P = 4, B = 192, lr_P = 0.02 * 4 = 0.08
        lr_p = LR0 * P_MAX
        torchrun(P_MAX, ["--mode", "ddp_train", "--threads", str(n_th), "--batch-size", str(B_LOCAL), "--lr", str(lr_p),
                         "--epochs", str(N_EP), "--out", os.path.join(RES, "step4_train_p4_scaled_lr.json")])

    # ---------- Крок 5: Контрольні точки K_c = 15 і відмова на s_f = 37 з --max-restarts=1 ----------
    ckpt_p = os.path.join(RES, "step5_ckpt.pt")
    marker_p = os.path.join(RES, "step5_fail.marker")
    out5_p = os.path.join(RES, "step5_recovery.json")
    for p in (ckpt_p, ckpt_p + ".tmp", marker_p, out5_p):
        if os.path.exists(p):
            os.remove(p)

    torchrun(
        P_MAX,
        [
            "--mode", "ddp_train",
            "--threads", str(n_th),
            "--batch-size", str(B_LOCAL),
            "--lr", str(LR0),
            "--epochs", str(N_EP),
            "--ckpt-every", str(K_C),
            "--ckpt-path", ckpt_p,
            "--fail-step", str(S_F),
            "--fail-marker", marker_p,
            "--out", out5_p,
        ],
        max_restarts=1,
    )
    print("All Lab 6 experiments finished successfully!")


if __name__ == "__main__":
    main()
