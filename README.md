# Лабораторна робота №6
## Розподілене навчання нейромережі за даними на CPU (Варіант 24)

- **Варіант:** 24
- **Датасет:** **Д6** (`OCTMNIST`, `octmnist.npz`, 4 класи, 1 канал, `N = 7000` train / `1000` test)
- **Ширина мережі:** **Ш3** (`c1 = 64, c2 = 128, h = 512`, `3 288 324` параметрів, `12.54 МіБ`)
- **Параметри:** `P_max = 4`, глобальний батч `B = 192` (`b = 48`), `K_c = 15`, `s_f = 37`, `n_th = ⌊8 / 4⌋ = 2` потоки на процес, бекенд `gloo`.

---

### Запуск у Docker (Linux CPU PyTorch — рекомендовано для Windows/WSL2)

1. Завантажити датасет у локальну теку `data/octmnist.npz` (або він автоматично довантажиться під час запуску):
   ```powershell
   py download_data.py
   ```
2. Зібрати Docker-образ із CPU-збіркою PyTorch:
   ```powershell
   docker build -t lab6-ddp:cpu .
   ```
3. Запустити повну серію експериментів (кроки 1–5) та обчислення таблиць і графіків:
   ```powershell
   docker run --rm --shm-size=2g -e HOST_C_PHYS=8 -e HOST_C_LOGICAL=12 -e HOST_RAM_GB=15.71 -v "${PWD}:/workspace" -w /workspace lab6-ddp:cpu python run_lab6.py
   docker run --rm -v "${PWD}:/workspace" -w /workspace lab6-ddp:cpu python analyze_lab6.py
   ```
   Щоб повторно запустити лише експеримент із відмовою та контрольною точкою (крок 5):
   ```powershell
   docker run --rm --shm-size=2g -e HOST_C_PHYS=8 -e HOST_C_LOGICAL=12 -e HOST_RAM_GB=15.71 -v "${PWD}:/workspace" -w /workspace lab6-ddp:cpu python run_lab6.py --only-step5
   ```

### Запуск напряму на Linux
```bash
python3 download_data.py
python3 run_lab6.py
python3 analyze_lab6.py
```

### Структура проєкту
- `model_data.py` — архітектура `Net`, формула `n_params`, завантаження `.npz` і обчислення `ACC` / `AUC`;
- `download_data.py` — завантаження `octmnist.npz` із Zenodo (через `curl` або `urllib`);
- `train_ddp.py` — режими однопроцесного навчання, ручної синхронізації, `DistributedDataParallel`, вимірів scaling та відновлення з контрольних точок;
- `run_lab6.py` — оркестратор прогонів (кроки 1–5);
- `analyze_lab6.py` — обчислення метрик та генерація таблиць і графіків у `results/analysis/`;
- [`REPORT.md`](REPORT.md) — повний звіт за лабораторною роботою.
