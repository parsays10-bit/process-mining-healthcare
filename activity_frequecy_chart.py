import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("UpFLux_Healthcare_Database_labeled.csv")

# -------------------------------------------------
# ACTIVITY FREQUENCY
# -------------------------------------------------

activity_counts = df["concept:name"].value_counts()

activity_percentage = (
    df["concept:name"]
    .value_counts(normalize=True) * 100
)

activity_table = pd.DataFrame({
    "Activity": activity_counts.index,
    "Count": activity_counts.values,
    "Percentage (%)": activity_percentage.values.round(2)
})

# -------------------------------------------------
# PLOT
# -------------------------------------------------

plt.figure()

activity_counts.plot(kind="bar")

plt.title("Activity Frequency")
plt.xlabel("Activity")
plt.ylabel("Number of events")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    "activity_frequency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------------------------------
# PRINT TABLE
# -------------------------------------------------

print("\nACTIVITY FREQUENCY TABLE")
print("------------------------")

print(activity_table)