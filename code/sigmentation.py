import pandas as pd

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("UpFLux_Healthcare_Database_labeled.csv")

# keep one row per case for segmentation attributes
case_attributes = df.groupby(
    "case:concept:name"
).agg({
    "Doença": "first",
    "Retorno": "first"
})

# -------------------------------------------------
# DIAGNOSIS DISTRIBUTION
# -------------------------------------------------

diagnosis_counts = case_attributes["Doença"].value_counts()

diagnosis_percentage = (
    case_attributes["Doença"]
    .value_counts(normalize=True) * 100
)

diagnosis_table = pd.DataFrame({
    "Diagnosis": diagnosis_counts.index,
    "Cases": diagnosis_counts.values,
    "Percentage (%)": diagnosis_percentage.values.round(2)
})

# -------------------------------------------------
# RETURN DISTRIBUTION
# -------------------------------------------------

return_counts = case_attributes["Retorno"].value_counts()

return_percentage = (
    case_attributes["Retorno"]
    .value_counts(normalize=True) * 100
)

return_table = pd.DataFrame({
    "Return type": return_counts.index,
    "Cases": return_counts.values,
    "Percentage (%)": return_percentage.values.round(2)
})

# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------

print("\nDIAGNOSIS DISTRIBUTION")
print("----------------------")
print(diagnosis_table)

print("\nRETURN DISTRIBUTION")
print("-------------------")
print(return_table)
