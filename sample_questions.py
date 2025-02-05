"""Sample questions and format datasets for forecasting.

Notes:
------

- Knowledge cutoff after 08.01
- Draw equally from 5 sources (ACLED, FRED, Wikipedia, dbnomics, yfinance)
- Include resolution criteria but not background

"""

import pandas as pd

# Load data
# dataset = "2024-12-08"
dataset = "2024-07-21"
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

# Remove rows with duplicate ids
df = df.sample(frac=1).drop_duplicates(subset="id")

# Add resolution criteria to question
df["question"] = df.apply(
    lambda row: row["question"]
    + "\n\nResolution Criteria: "
    + row["resolution_criteria"],
    axis=1,
)

df.groupby("source").size()


def sample_sources(df, source_counts):
    """Sample rows from a DataFrame with specific counts per source."""

    # Initialize empty DataFrame for results
    final_sample = pd.DataFrame()

    # Sample from each source according to specified counts
    for source, count in source_counts.items():
        source_df = df[df["source"] == source]

        # Check if we have enough samples
        available_samples = len(source_df)
        if available_samples < count:
            print(
                f"Warning: Requested {count} samples from {source} but only {available_samples} available"
            )
            sampled = source_df  # Take all available samples
        else:
            sampled = source_df.sample(n=count, random_state=42)

        final_sample = pd.concat([final_sample, sampled])

    return final_sample


# Define the desired counts per source
source_counts = {
    "acled": 12,
    "infer": 7,
    "manifold": 12,
    "metaculus": 12,
    "polymarket": 12,
    "dbnomics": 11,
    "fred": 11,
    "wikipedia": 12,
    "yfinance": 11,
}

df_sample = sample_sources(df, source_counts)

print(df_sample.groupby("source").size())
print(df_sample.groupby("resolution_date").size())
print(df_sample.groupby("resolved").size())
print(df_sample.groupby("resolved_to").size())


# Select the cols we need
df_sample = df_sample[["source", "id", "question", "resolution_date", "resolved_to"]]
df_sample["question_id"] = df_sample.apply(
    lambda row: f"{row['source']}_{row['id']}", axis=1
)
# Save to CSV
df_sample.to_csv(f"datasets/question_sample_{dataset}.csv", index=False)

# Merge prompts
prompts = pd.read_csv("prompts.csv")
prompts = prompts[["prompt_id", "Prompt Name", "Draft Prompt"]]

# Cartesian join, prompts to questions
df_sample["key"] = 0
prompts["key"] = 0
questions_prompts = pd.merge(df_sample, prompts, on="key")
questions_prompts.groupby("Prompt Name").size()

# String format question into prompt
questions_prompts["Prompt"] = questions_prompts.apply(
    lambda row: row["Draft Prompt"].format(Question=row["question"]), axis=1
)
# create question_id from source and id
questions_prompts["question_id"] = questions_prompts.apply(
    lambda row: f"{row['source']}_{row['id']}", axis=1
)

# Rename  & select cols
questions_prompts = questions_prompts.rename(columns={"Prompt": "prompt"})
questions_prompts = questions_prompts[["question_id", "prompt_id", "prompt"]]

# Save to CSV
questions_prompts.to_csv(f"datasets/questions_prompts_{dataset}.csv", index=False)


# Models
models = pd.DataFrame(
    {
        "model_name": [
            "claude-3-5-sonnet-20241022",
            "gpt-4o-2024-11-20",
            "claude-3-5-haiku-20241022",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        ],
        "model_provider": ["Anthropic", "OpenAI", "Anthropic", "TogetherAI"],
        "temperature": 0,
        "max_tokens": 2000,
    }
)

# Cartesian join, models to questions
questions_prompts["key"] = 0
models["key"] = 0
questions_models = pd.merge(questions_prompts, models, on="key")
questions_models.groupby("model_name").size()

# Create id columns
questions_models["id"] = questions_models.index

questions_models = questions_models[
    [
        "id",
        "question_id",
        "prompt_id",
        "prompt",
        "model_name",
        "model_provider",
        "temperature",
        "max_tokens",
    ]
]

# Save to CSV
questions_models.to_csv(f"datasets/questions_models_{dataset}.csv", index=False)
