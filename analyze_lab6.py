"""Аналіз результатів ЛР6 (Варіант 24) і побудова таблиць та графіків у results/analysis/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_data import N_EP

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "analysis")
os.makedirs(OUT, exist_ok=True)


def jload(name: str):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def fmt(v, spec=".4f", none_str="NaN (розходження)"):
    if v is None:
        return none_str
    return format(v, spec)


def main():
    env = jload("env.json")
    s1_tr = jload("step1_single_train.json")
    s1_bn = jload("step1_single_bench.json")
    s2_eq = jload("step2_equiv.json")
    s2_man = jload("step2_manual_bench.json")
    p_max = env["P_max"]
    s2_ddp = jload(f"step3_strong_p{p_max}.json")
    s2_ns = jload(f"step3_strong_nosync_p{p_max}.json")
    s2_tr = jload("step2_ddp_train.json")
    strong = []
    for p in range(1, p_max + 1):
        d = jload(f"step3_strong_p{p}.json")
        dns = jload(f"step3_strong_nosync_p{p}.json")
        strong.append((p, d, dns))

    x1_ddp = strong[0][1]["X"]
    strong_rows = []
    for p, d, dns in strong:
        t_step = d["t_step_ms"]
        t_ns = dns["t_step_ms"]
        t_comm = t_step - t_ns
        x_p = d["X"]
        s_p = x_p / x1_ddp
        e_p = s_p / p
        ef_p = None if p == 1 else ((1.0 / s_p - 1.0 / p) / (1.0 - 1.0 / p))
        c_p = t_comm / t_step
        strong_rows.append({
            "P": p,
            "b": d["b"],
            "t_step_ms": t_step,
            "t_ns_ms": t_ns,
            "t_comm_ms": round(t_comm, 3),
            "X": x_p,
            "S": round(s_p, 4),
            "E": round(e_p, 4),
            "e": round(ef_p, 4) if ef_p is not None else None,
            "c": round(c_p, 4),
        })

    # Крок 4: Weak scaling
    weak_list = [jload(f"step4_weak_p{p}.json") for p in range(1, p_max)] + [strong[-1][1]]
    x1_w = weak_list[0]["X"]
    weak_rows = []
    for d in weak_list:
        p = d["world_size"]
        x_p = d["X"]
        sw = x_p / x1_w
        ew = sw / p
        weak_rows.append({
            "P": p,
            "B_P": d["B"],
            "t_step_ms": d["t_step_ms"],
            "X": x_p,
            "S_w": round(sw, 4),
            "E_w": round(ew, 4),
        })

    s4_p1 = jload("step4_train_p1_b48.json")
    s4_sc = jload("step4_train_p4_scaled_lr.json")

    # Крок 5: Відновлення
    s5 = jload("step5_recovery.json")
    marker = jload("step5_fail.marker")
    t_f = float(marker["t_f"])
    ckpt_steps_before = [int(k) for k in s5["ckpt_times"].keys() if int(k) < env["s_f"]]
    last_ckpt_step = max(ckpt_steps_before)
    t_c = float(s5["ckpt_times"][str(last_ckpt_step)])
    t_first = float(s5["first_step_after_resume_unix"])
    t_r = t_first - t_f
    t_lost = (t_f - t_c) + t_r
    reexecuted = (env["s_f"] - 1) - last_ckpt_step

    # Таблиця 2 розрахунки
    t_comm_ddp_s2 = round(s2_ddp["t_step_ms"] - s2_ns["t_step_ms"], 3)
    c_ddp_s2 = round(t_comm_ddp_s2 / s2_ddp["t_step_ms"], 4)

    # ---- Графік 1: Strong scaling X(P) та P * X(1) ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ps = [r["P"] for r in strong_rows]
    xs_s = [r["X"] for r in strong_rows]
    ax.plot(ps, xs_s, "o-", color="C0", lw=2.0, label="Виміряне X(P) (DDP, B = 192)")
    ax.plot(ps, [p * x1_ddp for p in ps], "--", color="gray", lw=1.3, label="Ідеальне лінійне P · X(1)")
    for p, xv, r in zip(ps, xs_s, strong_rows):
        ax.annotate(f"{xv:.0f}\n(S={r['S']:.2f})", (p, xv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(ps)
    ax.set_ylim(0, max(max(xs_s), p_max * x1_ddp) * 1.15)
    ax.set_xlabel("Кількість процесів P")
    ax.set_ylabel("Пропускна здатність X, прикл./с")
    ax.set_title("ЛР6 (Варіант 24, OCTMNIST Ш3): Strong scaling (B = 192)", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "strong_scaling.png"), dpi=130)
    plt.close(fig)

    # ---- Графік 2: Weak scaling X(P) та E_w(P) ----
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
    pw = [r["P"] for r in weak_rows]
    xw = [r["X"] for r in weak_rows]
    ew = [r["E_w"] for r in weak_rows]
    axs[0].plot(pw, xw, "s-", color="C1", lw=1.8, label="Виміряне X(P) (b = 48)")
    axs[0].plot(pw, [p * x1_w for p in pw], "--", color="gray", label="Ідеальне P · X_w(1)")
    axs[0].set_xticks(pw)
    axs[0].set_ylim(0, max(max(xw), p_max * x1_w) * 1.15)
    axs[0].set_xlabel("P, процесів")
    axs[0].set_ylabel("X(P), прикл./с")
    axs[0].set_title("Weak scaling: пропускна здатність", fontsize=10.5)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="upper left")

    axs[1].plot(pw, ew, "o-", color="C2", lw=1.8, label="Ефективність E_w(P)")
    axs[1].axhline(1.0, color="gray", ls="--", lw=1.0)
    axs[1].set_xticks(pw)
    axs[1].set_ylim(0, 1.18)
    axs[1].set_xlabel("P, процесів")
    axs[1].set_ylabel("E_w = S_w / P")
    axs[1].set_title("Weak scaling: ефективність", fontsize=10.5)
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "weak_scaling.png"), dpi=130)
    plt.close(fig)

    # ---- Графік 3: Порівняння втрат на кроках навколо s_f = 37 ----
    sf = env["s_f"]
    steps_show = list(range(1, s2_tr["total_steps"] + 1))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axs[0].plot(steps_show, [s2_tr["losses_by_step"][str(k)] for k in steps_show], "-", color="C0", lw=1.4, label="DDP без відмови")
    axs[0].plot(steps_show, [s5["losses_by_step"][str(k)] for k in steps_show], "--", color="C1", lw=1.3, label="DDP з відмовою на s_f=37")
    axs[0].axvline(last_ckpt_step, color="green", ls=":", lw=1.1, label=f"Контрольна точка ({last_ckpt_step})")
    axs[0].axvline(sf, color="red", ls="--", lw=1.1, label=f"Відмова s_f = {sf}")
    axs[0].set_xlabel("Крок оптимізатора")
    axs[0].set_ylabel("Втрата (CrossEntropyLoss)")
    axs[0].set_title("Траєкторія навчання за 3 епохи (108 кроків)", fontsize=10.5)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=7.5)

    win_k = list(range(sf - 2, sf + 3))
    axs[1].plot(win_k, [s2_tr["losses_by_step"][str(k)] for k in win_k], "o-", color="C0", label="Без відмови")
    axs[1].plot(win_k, [s5["losses_by_step"][str(k)] for k in win_k], "x--", ms=8, color="C1", label="З відмовою і відновленням")
    axs[1].axvline(sf, color="red", ls=":", lw=1.0)
    axs[1].set_xticks(win_k)
    axs[1].set_xlabel("Крок оптимізатора")
    axs[1].set_ylabel("Втрата")
    axs[1].set_title(f"Збіг втрат біля кроку відмови s_f = {sf}", fontsize=10.5)
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_recovery.png"), dpi=130)
    plt.close(fig)

    loss_diffs = [
        abs(s5["losses_by_step"][str(k)] - s2_tr["losses_by_step"][str(k)])
        for k in win_k
        if s5["losses_by_step"][str(k)] is not None and s2_tr["losses_by_step"][str(k)] is not None
    ]
    max_abs_loss_diff_window = max(loss_diffs) if len(loss_diffs) == len(win_k) else None

    summary = {
        "env": env,
        "step1": {"train": s1_tr, "bench": s1_bn},
        "step2": {"equiv": s2_eq, "manual_bench": s2_man, "ddp_bench": s2_ddp, "ddp_nosync": s2_ns,
                  "t_comm_exp_ms": t_comm_ddp_s2, "c": c_ddp_s2, "ddp_train": {k: v for k, v in s2_tr.items() if k != "losses_by_step"}},
        "step3_strong": strong_rows,
        "step4_weak": weak_rows,
        "step4_quality": {
            "p1_b48": {k: v for k, v in s4_p1.items() if k != "losses_by_step"},
            "p4_b192_lr0": {k: v for k, v in s2_tr.items() if k != "losses_by_step"},
            "p4_b192_scaled_lr": {k: v for k, v in s4_sc.items() if k != "losses_by_step"},
        },
        "step5_recovery": {
            "K_c": env["K_c"],
            "s_f": sf,
            "last_ckpt_step": last_ckpt_step,
            "reexecuted_steps": reexecuted,
            "t_r_s": round(t_r, 3),
            "t_lost_s": round(t_lost, 3),
            "window_losses_fail": {str(k): s5["losses_by_step"][str(k)] for k in win_k},
            "window_losses_clean": {str(k): s2_tr["losses_by_step"][str(k)] for k in win_k},
            "max_abs_loss_diff_window": round(max_abs_loss_diff_window, 7) if max_abs_loss_diff_window is not None else None,
            "acc_fail": s5["acc"],
            "acc_clean": s2_tr["acc"],
        },
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = [
        "# Зведені таблиці ЛР6 (Варіант 24, Д6 OCTMNIST, Ш3)\n",
        "## Таблиця 1. Стенд\n",
        "| Процесор | Фізичних ядер C_phys | Логічних ядер | Памʼять, ГБ | ОС | Python | PyTorch | n_th |",
        "|---|---|---|---|---|---|---|---|",
        f"| {env['cpu']} | {env['c_phys']} | {env['c_logical']} | {env['ram_gb']} | {env['os']} | {env['python']} | {env['pytorch']} | {env['n_th']} |\n",
        "| Датасет | K | C | c1 / c2 / h | n_params | M, МіБ | N | B | P_max |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {env['dataset']} | {env['K']} | {env['C']} | {env['widths']} | {env['n_params']} | {env['M_mib']:.2f} | {env['N']} | {env['B']} | {env['P_max']} |\n",
        "## Таблиця 2. Синхронізація градієнтів\n",
        "| Режим | t_step, мс | Час обміну на крок, мс | Частка кроку в обміні | X, прикл./с |",
        "|---|---|---|---|---|",
        f"| 1 процес, батч B | {s1_bn['t_step_ms']:.2f} | 0 | 0 | {s1_bn['X']:.2f} |",
        f"| {p_max} процесів, ручна синхронізація | {s2_man['t_step_ms']:.2f} | {s2_man['t_sync_ms']:.2f} | {s2_man['comm_share']:.4f} | {s2_man['X']:.2f} |",
        f"| {p_max} процесів, DDP | {s2_ddp['t_step_ms']:.2f} | {t_comm_ddp_s2:.2f} | {c_ddp_s2:.4f} | {s2_ddp['X']:.2f} |",
        f"| {p_max} процесів, DDP, no_sync | {s2_ns['t_step_ms']:.2f} | 0 | 0 | {s2_ns['X']:.2f} |\n",
        "| max \\|g − ḡ\\| / max \\|g\\| | Різниця сум ваг між процесами після 50 кроків | ACC, 1 процес | ACC, DDP на P_max | AUC, 1 процес | AUC, DDP на P_max |",
        "|---|---|---|---|---|---|",
        f"| {s2_eq['max_grad_rel_diff']:.3e} | {s2_eq['weight_sum_diff_50_steps']:.3e} | {s1_tr['acc']:.4f} | {s2_tr['acc']:.4f} | {s1_tr['auc']:.4f} | {s2_tr['auc']:.4f} |\n",
        "## Таблиця 3. Strong scaling, B = 192 const\n",
        "| P | b | t_step, мс | t_step,ns, мс | X, прикл./с | S | E | e | c |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in strong_rows:
        e_str = "−" if r["e"] is None else f"{r['e']:.4f}"
        md.append(f"| {r['P']} | {r['b']} | {r['t_step_ms']:.2f} | {r['t_ns_ms']:.2f} | {r['X']:.2f} | {r['S']:.4f} | {r['E']:.4f} | {e_str} | {r['c']:.4f} |")

    md.extend([
        "\n## Таблиця 4. Weak scaling, b = 48 const\n",
        "| P | B_P | t_step, мс | X, прикл./с | S_w | E_w |",
        "|---|---|---|---|---|---|",
    ])
    for r in weak_rows:
        md.append(f"| {r['P']} | {r['B_P']} | {r['t_step_ms']:.2f} | {r['X']:.2f} | {r['S_w']:.4f} | {r['E_w']:.4f} |")

    md.extend([
        "\n## Таблиця 5. Якість за різних глобальних батчів\n",
        "| Режим | B_P | η | Кроків оптимізатора за n_ep епох | ACC | AUC | Найбільша втрата за останню епоху |",
        "|---|---|---|---|---|---|---|",
        f"| P = 1 | {s4_p1['B']} | {s4_p1['lr']} | {s4_p1['total_steps']} | {fmt(s4_p1['acc'])} | {fmt(s4_p1['auc'])} | {fmt(s4_p1['max_loss_last_epoch'])} |",
        f"| P = {p_max}, з кроку 2 | {s2_tr['B']} | {s2_tr['lr']} | {s2_tr['total_steps']} | {fmt(s2_tr['acc'])} | {fmt(s2_tr['auc'])} | {fmt(s2_tr['max_loss_last_epoch'])} |",
        f"| P = {p_max} | {s4_sc['B']} | {s4_sc['lr']} | {s4_sc['total_steps']} | {fmt(s4_sc['acc'])} | {fmt(s4_sc['auc'])} | {fmt(s4_sc['max_loss_last_epoch'])} |\n",
        "## Таблиця 6. Відмова і відновлення\n",
        "| K_c | s_f | Крок останньої контрольної точки | Кроків виконано повторно | t_lost, с |",
        "|---|---|---|---|---|",
        f"| {env['K_c']} | {sf} | {last_ckpt_step} | {reexecuted} | {t_lost:.2f} (t_r={t_r:.2f} с) |\n",
        f"| Крок | {sf-2} | {sf-1} | {sf} | {sf+1} | {sf+2} | ACC після {N_EP} епох |",
        "|---|---|---|---|---|---|---|",
        "| Втрата, прогін з відмовою | " + " | ".join(fmt(s5['losses_by_step'][str(k)]) for k in win_k) + f" | {fmt(s5['acc'])} |",
        "| Втрата, прогін без відмови | " + " | ".join(fmt(s2_tr['losses_by_step'][str(k)]) for k in win_k) + f" | {fmt(s2_tr['acc'])} |",
    ])
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("Lab 6 analysis written to results/analysis/")


if __name__ == "__main__":
    main()
