"""Sample questions and format datasets for forecasting.

Notes:
------

- Knowledge cutoff after 08.01
- Draw equally from 5 sources (ACLED, FRED, Wikipedia, dbnomics, yfinance)
- Include resolution criteria but not background

"""

import pandas as pd

# Load data
dataset = "2024-12-08"
# dataset = "2024-07-21"
df = pd.read_csv(f"datasets/{dataset}_merged.csv")
df = df.dropna(subset=["question"])

# filter by resolution_date
df = df[df.resolution_date > "2024-08-01"]
df.groupby("source").size()

df.groupby("resolution_date").size()

# Substitute "{resolution_date}" with the resolution_date col
df = df.dropna(subset=["question"])
df["question"] = df.apply(
    lambda row: row["question"]
    .replace("{resolution_date}", str(row["resolution_date"]))
    .replace("{forecast_due_date}", str(row["forecast_due_date"])),
    axis=1,
)

# Add resolution criteria to question
df["question"] = df.apply(
    lambda row: row["question"]
    + "\n\nResolution Criteria: "
    + row["resolution_criteria"],
    axis=1,
)

# Sample evenly from each category
# df_sample = df.groupby("source").sample(25).reset_index(drop=True)

# Sample 100 randomly
df_sample = df.sample(100).reset_index(drop=True)

df_sample.groupby("source").size()
df_sample.groupby("resolution_date").size()
df_sample.groupby("resolved").size()
df_sample.groupby("resolved_to").size()


# Select the cols we need
df_sample = df_sample[["source", "id", "question", "resolution_date", "resolved_to"]]
# Save to CSV
df_sample.to_csv(f"datasets/question_sample_{dataset}.csv", index=False)
