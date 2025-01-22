"""Convert ForecastBench JSON data to CSV format.

Setup: clone the ForecastBench repo and into a directory called forecasting-datasets.

Notes/problems:
- Some questions have multiple ids (i.e. the question links to 2 responses).
I haven't dealt with these yet. They appear to be combinations of existing questions,
I don't know if/how much we care about these.

- I'm removing non-resolved questions from the resolution dataset. I'm assuming these
are showing the current market value on pred markets.

- I'm merging on id and resolution date for q's with multiple resolution dates.

"""

import os
import json
from typing import Dict, List, Any

import pandas as pd


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load and return JSON data from a file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def process_questions(questions_data: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Process questions data, handling multiple resolution dates."""
    # First, create a list to store expanded questions
    expanded_questions = []

    forecast_due_date = questions_data["forecast_due_date"]
    for question in questions_data["questions"]:
        # Skip if id is not a string
        if not isinstance(question.get("id"), str):
            continue

        # Look for resolution date in resolution_dates > resolution_date > market_info_close_datetime
        resolution_dates = question.get("resolution_dates") or question.get(
            "resolution_date"
        )
        if resolution_dates == "N/A":
            resolution_dates = question.get("market_info_close_datetime")
            # trim datetime to date
            if resolution_dates:
                resolution_dates = resolution_dates.split("T")[0]

        # Handle both single string and list of resolution dates
        if isinstance(resolution_dates, str):
            resolution_dates = [resolution_dates]
        elif not isinstance(resolution_dates, list):
            continue

        # Create a separate entry for each resolution date
        for resolution_date in resolution_dates:
            question_copy = question.copy()
            # question_copy["resolution_date"] = resolution_date
            question_copy["forecast_due_date"] = forecast_due_date
            expanded_questions.append(question_copy)

    # Convert to DataFrame
    questions_df = pd.DataFrame(expanded_questions)

    # Drop the original resolution_dates column since we've expanded it
    if "resolution_dates" in questions_df.columns:
        questions_df = questions_df.drop("resolution_dates", axis=1)

    return questions_df


def process_resolutions(resolutions_data: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Process resolutions data."""
    resolutions_df = pd.DataFrame(resolutions_data["resolutions"])

    # Filter out non-string IDs
    resolutions_df = resolutions_df[
        resolutions_df["id"].apply(lambda x: isinstance(x, str))
    ]
    # Filter out unresolved questions
    resolutions_df = resolutions_df[resolutions_df["resolved"]]

    return resolutions_df


def merge_data(
    questions_df: pd.DataFrame, resolutions_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge questions and resolutions data."""
    # Merge on both id and resolution_date
    merged_df = pd.merge(
        questions_df,
        resolutions_df,
        on=["id", "source"],
        how="right",
    )

    return merged_df


def main():
    """Load question and resolution JSONs, merge them, and save to CSV."""
    # Configuration
    load_dir = "../forecasting-datasets/datasets/"
    # question_set = "2024-12-08"
    question_set = "2024-07-21"
    save_dir = "datasets/"

    # File paths
    questions_json_file = os.path.join(
        load_dir, "question_sets", f"{question_set}-llm.json"
    )
    resolutions_json_file = os.path.join(
        load_dir, "resolution_sets", f"{question_set}_resolution_set.json"
    )
    output_csv_file = os.path.join(save_dir, f"{question_set}_merged.csv")

    # Load data
    questions_data = load_json_file(questions_json_file)
    resolutions_data = load_json_file(resolutions_json_file)

    # Process data
    questions_df = process_questions(questions_data)
    resolutions_df = process_resolutions(resolutions_data)

    # Merge data
    merged_df = merge_data(questions_df, resolutions_df)

    # Save to CSV
    merged_df.to_csv(output_csv_file, index=False)
    print(f"Processed {len(merged_df)} rows")
    print(f"Data saved to {output_csv_file}")


if __name__ == "__main__":
    main()
