import os
import gc
import optuna
from concurrent.futures import ProcessPoolExecutor
from main import run_walk_forward
from main import CSV_PATH
from get_data import FetchStocks
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

run_macro = False
run_all = False

base_rebalance = 30
base_lookback = 504
base_entry_z1 = 2.9
base_entry_z2 = 4.2
base_stop_z = 4.7
base_exit_z = 0.9
base_t1_weight = 0.75

start_date = "2023-01-01"
end_date = "2024-12-31"

study_name = "micros_2.0"
trials = 2000
workers = 14

optuna.logging.set_verbosity(optuna.logging.INFO)

print("Loading data")
GLOBAL_MASTER = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
GLOBAL_TICKERS = FetchStocks(str(CSV_PATH)).get_all_tickers()
print("Data loaded. Starting...")

def macro(trial):

    rebalance_step = trial.suggest_int('rebalance_step', 15, 90, step=5)

    lookback = trial.suggest_int('lookback', 189, 504, step=21)

    return {
        "rebalance_step": rebalance_step,  
        "lookback": lookback,  
        "entry_z1": base_entry_z1,  
        "entry_z2": base_entry_z2,  
        "stop_z": base_stop_z,  
        "exit_z": base_exit_z,  
        "t1_weight": base_t1_weight,  
        "start_date": start_date,
        "end_date": end_date
    }

def micro(trial):
    entry_z1 = trial.suggest_float('entry_z1', 1.0, 3.0, step=0.1)

    z2_gap = trial.suggest_float('z2_gap', 0.2, 1.5, step=0.1)
    entry_z2 = entry_z1 + z2_gap

    stop_gap = trial.suggest_float('stop_gap', 0.5, 2.0, step=0.1)
    stop_z = entry_z2 + stop_gap

    exit_z = trial.suggest_float('exit_z', 0.0, 1.5, step=0.1)
    t1_weight = trial.suggest_float('t1_weight', 0.6, 1.0, step=0.1)

    return {
        "rebalance_step": base_rebalance,  
        "lookback": base_lookback,  
        "entry_z1": entry_z1,  
        "entry_z2": entry_z2,  
        "stop_z": stop_z,  
        "exit_z": exit_z,  
        "t1_weight": t1_weight,  
        "start_date": start_date,
        "end_date": end_date
    }


def mnm(trial):
    rebalance_step = trial.suggest_int('rebalance_step', 15, 90, step=5)
    lookback = trial.suggest_int('lookback', 189, 1008, step=21)

    entry_z1 = trial.suggest_float('entry_z1', 1.5, 3.5, step=0.1)

    z2_gap = trial.suggest_float('z2_gap', 0.3, 2.0, step=0.1)
    entry_z2 = entry_z1 + z2_gap

    stop_gap = trial.suggest_float('stop_gap', 0.5, 2.0, step=0.1)
    stop_z = entry_z2 + stop_gap

    exit_z = trial.suggest_float('exit_z', 0.0, 1.5, step=0.1)
    t1_weight = trial.suggest_float('t1_weight', 0.5, 1.0, step=0.05)

    return {
        "rebalance_step": rebalance_step,
        "lookback": lookback,
        "entry_z1": entry_z1,
        "entry_z2": entry_z2,
        "stop_z": stop_z,
        "exit_z": exit_z,
        "t1_weight": t1_weight,
        "start_date": start_date,
        "end_date": end_date
    }


def objective(trial):
    try:
        if run_all:
            params = mnm(trial)
        elif run_macro:
            params = macro(trial)
        else:
            params = micro(trial)

        try:
            sharpe, max_dd = run_walk_forward(params, df_master=GLOBAL_MASTER, df_tickers=GLOBAL_TICKERS)
        except Exception as e:
            return -5.0

        DRAWDOWN_LIMIT = -0.10

        if max_dd < DRAWDOWN_LIMIT:
            excess_dd = abs(max_dd) - abs(DRAWDOWN_LIMIT)
            penalty = excess_dd * 20.0
            return sharpe - penalty

        return sharpe

    finally:
        gc.collect()


def run_worker_process(_):
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_study.db",
        engine_kwargs={"connect_args": {"timeout": 60.0}}
    )

    study = optuna.load_study(
        study_name=study_name,
        storage=storage
    )

    tpw = int(trials/workers)
    study.optimize(objective, n_trials=tpw, n_jobs=1)


if __name__ == "__main__":
    print("Starting optimization...")

    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_study.db",
        engine_kwargs={"connect_args": {"timeout": 60.0}}
    )

    optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize"
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(run_worker_process, range(workers))

    study = optuna.load_study(study_name=study_name, storage=storage)

    print("\n" + "=" * 40)
    print("OPTIMIZATION COMPLETE")
    print("=" * 40)
    print(f"Best Net Sharpe Ratio: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
