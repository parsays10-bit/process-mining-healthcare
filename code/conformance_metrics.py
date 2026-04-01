import pandas as pd
import pm4py

# -------------------------------------------------
# LOAD EVENT LOG
# -------------------------------------------------

log_df = pd.read_csv("UpFLux_Healthcare_Database_labeled.csv")

log_df = log_df.rename(columns={
    "case:concept:name": "case_id",
    "concept:name": "activity",
    "time:timestamp": "timestamp"
})

log_df["timestamp"] = pd.to_datetime(
    log_df["timestamp"],
    errors="coerce"
)

log_df = log_df.dropna(subset=[
    "case_id",
    "activity",
    "timestamp"
])

# -------------------------------------------------
# FORMAT FOR PM4PY
# -------------------------------------------------

log_df = pm4py.format_dataframe(
    log_df,
    case_id="case_id",
    activity_key="activity",
    timestamp_key="timestamp"
)

event_log = pm4py.convert_to_event_log(log_df)

# -------------------------------------------------
# DISCOVER PROCESS MODELS
# -------------------------------------------------

models = {}

models["Alpha Miner"] = pm4py.discover_petri_net_alpha(event_log)

models["Heuristic Miner"] = pm4py.discover_petri_net_heuristics(event_log)

models["Inductive Miner"] = pm4py.discover_petri_net_inductive(event_log)

# -------------------------------------------------
# MODEL QUALITY METRICS
# -------------------------------------------------

results = []

for model_name, (net, im, fm) in models.items():

    fitness = pm4py.fitness_token_based_replay(
        event_log,
        net,
        im,
        fm
    )["log_fitness"]

    precision = pm4py.precision_token_based_replay(
        event_log,
        net,
        im,
        fm
    )

    results.append({
        "Model": model_name,
        "Fitness": fitness,
        "Precision": precision
    })

# -------------------------------------------------
# RESULTS TABLE
# -------------------------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL QUALITY METRICS")
print("---------------------")

print(results_df)

# -------------------------------------------------
# SORT BY FITNESS
# -------------------------------------------------

print("\nSORTED BY FITNESS")
print("-----------------")

print(
    results_df
    .sort_values(by="Fitness", ascending=False)
)

# -------------------------------------------------
# SORT BY PRECISION
# -------------------------------------------------

print("\nSORTED BY PRECISION")
print("-------------------")

print(
    results_df
    .sort_values(by="Precision", ascending=False)
)
