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


def stratified_sample(df, n_samples=100, size_threshold=15):
    """
    Sample rows from a DataFrame with complete sampling for small sources and
    even sampling across larger sources.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with a 'source' column
    n_samples : int, default=100
        Total number of desired samples
    size_threshold : int, default=15
        Threshold for considering a source as "small"

    Returns:
    --------
    pandas.DataFrame
        Sampled DataFrame with the desired number of rows
    """
    # Get source sizes
    source_sizes = df.groupby("source").size()

    # Separate small and large sources
    small_sources = source_sizes[source_sizes < size_threshold].index
    large_sources = source_sizes[source_sizes >= size_threshold].index

    # Take all samples from small sources
    small_samples = df[df["source"].isin(small_sources)]
    n_small_samples = len(small_samples)

    # Calculate remaining samples needed
    remaining_samples = n_samples - n_small_samples

    # Calculate even distribution for large sources
    samples_per_large_source = remaining_samples // len(large_sources)
    extra_samples = remaining_samples % len(large_sources)

    # Sample from large sources
    large_samples = pd.DataFrame()
    for i, source in enumerate(large_sources):
        # Add an extra sample to the first 'extra_samples' sources to distribute remainder
        n_source_samples = samples_per_large_source + (1 if i < extra_samples else 0)
        source_df = df[df["source"] == source]
        sampled = source_df.sample(n=n_source_samples, random_state=42)
        large_samples = pd.concat([large_samples, sampled])

    # Combine small and large samples
    final_sample = pd.concat([small_samples, large_samples])

    # Verify total samples
    actual_samples = len(final_sample)
    if actual_samples != n_samples:
        print(
            f"Warning: Returned {actual_samples} samples instead of {n_samples} due to rounding"
        )

    return final_sample


df_sample = stratified_sample(df)

print(df_sample.groupby("source").size())
print(df_sample.groupby("resolution_date").size())
print(df_sample.groupby("resolved").size())
print(df_sample.groupby("resolved_to").size())


# Select the cols we need
df_sample = df_sample[["source", "id", "question", "resolution_date", "resolved_to"]]
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
