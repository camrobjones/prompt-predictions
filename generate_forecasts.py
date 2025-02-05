import pandas as pd
import openai
import re

client = openai.OpenAI()
dataset = "2024-07-21"

df = pd.read_csv(f"datasets/questions_models_{dataset}.csv")

df = df[df.model_name == "gpt-4o-2024-11-20"]

row = df.iloc[0]


def generate_response(prompt, model_name, temperature=0, max_tokens=2000):

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )
    content = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens
    return (content, output_tokens)


def extract_forecast(response):
    forecasts = re.findall("Forecast: (\d*\.?\d+)%", response)

    if forecasts:
        forecast = float(forecasts[-1]) / 100
    else:
        forecast = "No forecast given"

    return (forecast, len(forecasts))


# Randomly sample one row of df for each value of question_id

df.groupby(["question_id", "prompt_id"]).size()


def balanced_sample(df, random_state=42):
    """
    Sample rows to ensure:
    1. Exactly one row per question_id (100 total rows)
    2. At least one row for each prompt_id
    3. Random distribution of prompt_ids

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with 'question_id' and 'prompt_id' columns
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        Sampled DataFrame meeting the criteria
    """
    import pandas as pd
    import numpy as np

    # Set random seed
    np.random.seed(random_state)

    # Get all unique prompt_ids and question_ids
    all_prompt_ids = list(df["prompt_id"].unique())
    all_question_ids = list(df["question_id"].unique())

    # Initialize tracking sets
    selected_question_ids = set()
    covered_prompt_ids = set()
    rows_list = []

    # First, ensure we get at least one of each prompt_id
    np.random.shuffle(all_prompt_ids)  # Randomize order
    for prompt_id in all_prompt_ids:
        if prompt_id in covered_prompt_ids:
            continue

        # Get available question_ids for this prompt_id
        available_questions = df[
            (df["prompt_id"] == prompt_id)
            & (~df["question_id"].isin(selected_question_ids))
        ]["question_id"].unique()

        if len(available_questions) > 0:
            # Select a random question_id
            selected_question = np.random.choice(available_questions)

            # Get one random row for this combination
            selected_row = df[
                (df["question_id"] == selected_question)
                & (df["prompt_id"] == prompt_id)
            ].iloc[0]

            rows_list.append(selected_row)
            selected_question_ids.add(selected_question)
            covered_prompt_ids.add(prompt_id)

    # Get remaining question_ids to cover
    remaining_question_ids = set(all_question_ids) - selected_question_ids

    # For remaining question_ids, sample randomly from any prompt_id
    for question_id in remaining_question_ids:
        question_rows = df[df["question_id"] == question_id]
        selected_row = question_rows.sample(
            n=1, random_state=np.random.randint(1000)
        ).iloc[0]
        rows_list.append(selected_row)

    # Create final DataFrame
    result = pd.DataFrame(rows_list)

    # Print distribution stats
    print("\nSampling Results:")
    print("-----------------")
    print(f"Total rows: {len(result)}")
    print(f"Unique question_ids: {result['question_id'].nunique()}")
    print(f"Unique prompt_ids: {result['prompt_id'].nunique()}")
    print("\nPrompt ID distribution:")
    print(result["prompt_id"].value_counts().describe())

    # Verify constraints
    assert len(result) == len(all_question_ids), "Wrong number of total rows"
    assert set(result["question_id"]) == set(
        all_question_ids
    ), "Missing some question_ids"
    assert set(result["prompt_id"]).issuperset(
        set(all_prompt_ids)
    ), "Missing some prompt_ids"

    return result


# Sample the data
df_balanced = balanced_sample(df)
df_balanced.groupby("prompt_id").size()
df_balanced.groupby("question_id").size()


response_data = []

for i, row in df_balanced.iterrows():

    question_id = row["question_id"]
    if question_id in response_df.question_id.to_list():
        print("question already done")
        continue

    prompt = row["prompt"]

    model_name = row["model_name"]

    print("\n\n" + "=" * 30 + "\n")
    print(prompt)
    print("-" * 30 + "\n")

    response, tokens = generate_response(prompt, model_name)

    print(f"Response: {response}")

    forecast, n_forecasts = extract_forecast(response)

    print(f"Tokens used: {tokens}")
    print(f"N. forecasts: {n_forecasts}")
    print(f"Forecast: {forecast}")

    row = dict(row)
    row["response"] = response
    row["tokens"] = tokens
    row["n_forecasts"] = n_forecasts
    row["forecast"] = forecast

    response_data.append(row)

    response_df = pd.DataFrame(response_data)
