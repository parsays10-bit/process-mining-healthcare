import pandas as pd

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("UpFLux_Healthcare_Database_labeled.csv")

df["time:timestamp"] = pd.to_datetime(
    df["time:timestamp"],
    errors="coerce"
)

df = df.dropna(subset=[
    "case:concept:name",
    "concept:name",
    "time:timestamp"
])

# -------------------------------------------------
# BASIC LOG INFORMATION
# -------------------------------------------------

print("\nLOG SUMMARY")
print("-----------")

print("Number of events:", len(df))
print("Number of cases:", df["case:concept:name"].nunique())
print("Number of activities:", df["concept:name"].nunique())

# -------------------------------------------------
# EVENTS PER CASE (CASE SIZE KPI)
# -------------------------------------------------

events_per_case = df.groupby(
    "case:concept:name"
)["concept:name"].count()

print("\nEVENTS PER CASE KPI")
print("-------------------")

print("Min events per case:", events_per_case.min())
print("Max events per case:", events_per_case.max())
print("Mean events per case:", events_per_case.mean())
print("Median events per case:", events_per_case.median())

print("\nDistribution of events per case:")
print(events_per_case.value_counts().sort_index())

# -------------------------------------------------
# CASE DURATION KPI
# -------------------------------------------------

case_duration = df.groupby(
    "case:concept:name"
)["time:timestamp"].agg(["min", "max"])

case_duration["duration"] = (
    case_duration["max"] - case_duration["min"]
)

print("\nCASE DURATION KPI")
print("-----------------")

print("Min duration:", case_duration["duration"].min())
print("Max duration:", case_duration["duration"].max())
print("Mean duration:", case_duration["duration"].mean())
print("Median duration:", case_duration["duration"].median())

# -------------------------------------------------
# LONGEST CASE TRACE
# -------------------------------------------------

longest_case = events_per_case.idxmax()

print("\nCASE WITH MAX EVENTS:", longest_case)

print("\nTrace of longest case:")

trace = df[df["case:concept:name"] == longest_case]

print(
    trace
    .sort_values("time:timestamp")[
        ["concept:name", "time:timestamp"]
    ]
)

# -------------------------------------------------
# REWORK KPI
# -------------------------------------------------

def detect_rework(group):
    return group["concept:name"].duplicated().any()

rework_cases = df.groupby(
    "case:concept:name"
).apply(detect_rework)

rework_percentage = rework_cases.mean() * 100

print("\nREWORK KPI")
print("----------")

print("Cases with rework:", rework_cases.sum())
print("Rework percentage:", round(rework_percentage, 2))

# -------------------------------------------------
# KPI PER CASE TABLE
# -------------------------------------------------

kpi_per_case = pd.DataFrame({
    "case_size": events_per_case,
    "duration": case_duration["duration"],
    "rework": rework_cases
})

case_attributes = df.groupby(
    "case:concept:name"
).agg({
    "Doença": "first",
    "Retorno": "first"
})

kpi_per_case = kpi_per_case.join(case_attributes)

# -------------------------------------------------
# KPI BY DIAGNOSIS
# -------------------------------------------------

print("\nKPI BY DIAGNOSIS")
print("----------------")

diagnosis_kpi = kpi_per_case.groupby(
    "Doença"
).agg({
    "case_size": "mean",
    "duration": "mean",
    "rework": "mean"
})

print(diagnosis_kpi)

# -------------------------------------------------
# KPI BY DISCHARGE TYPE
# -------------------------------------------------

print("\nKPI BY DISCHARGE TYPE")
print("---------------------")

retorno_kpi = kpi_per_case.groupby(
    "Retorno"
).agg({
    "case_size": "mean",
    "duration": "mean",
    "rework": "mean"
})

print(retorno_kpi)

# -------------------------------------------------
# OUTLIER DETECTION
# -------------------------------------------------

size_threshold = events_per_case.quantile(0.99)
duration_threshold = case_duration["duration"].quantile(0.99)

large_cases = events_per_case[
    events_per_case >= size_threshold
]

long_cases = case_duration[
    case_duration["duration"] >= duration_threshold
]

rework_case_ids = rework_cases[
    rework_cases == True
]

print("\nOUTLIER CASES")
print("-------------")

print("\nLarge case size:")
print(large_cases)

print("\nLong duration:")
print(long_cases["duration"])

print("\nCases with rework:")
print(rework_case_ids.head(10))

# -------------------------------------------------
# ACTIVITY FREQUENCY TABLE
# -------------------------------------------------

activity_frequency = df["concept:name"].value_counts()

activity_percentage = (
    df["concept:name"]
    .value_counts(normalize=True) * 100
)

activity_table = pd.DataFrame({
    "Activity": activity_frequency.index,
    "Count": activity_frequency.values,
    "Percentage (%)": activity_percentage.values.round(2)
})

activity_table = activity_table.sort_values(
    by="Count",
    ascending=False
)

print("\nACTIVITY FREQUENCY TABLE")
print("------------------------")

print(activity_table)

# -------------------------------------------------
# TOP 10 LONGEST CASES
# -------------------------------------------------

print("\nTOP 10 LONGEST CASES")
print("--------------------")

print(
    case_duration
    .sort_values("duration", ascending=False)
    .head(10)
)
