import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_forecast_accuracy():
    """
    Analyze the accuracy of forecasting models across different prompts by:
    1. Loading and preprocessing datasets
    2. Extracting forecasts from model responses
    3. Matching responses to questions and prompts
    4. Calculating Brier scores
    5. Analyzing performance by prompt type
    6. Visualizing the results
    """
    print("Starting analysis of model forecast accuracy...")

    # 1. Load datasets
    print("Loading datasets...")
    questions_sample = pd.read_csv("datasets/question_sample_2024-12-08.csv")
    model_questions = pd.read_csv("datasets/questions_models_2024-12-08.csv")
    sonnet = pd.read_csv("data/sonnet.csv")

    print(
        f"Loaded {len(questions_sample)} questions, {len(model_questions)} model-question pairs, and {len(sonnet)} sonnet responses"
    )

    # 2. Extract forecasts from model responses
    def extract_forecast(response):
        if not isinstance(response, str):
            return None

        forecasts = re.findall(r"Forecast:[\s\*]*(\d*\.?\d+)%", response)
        if forecasts:
            # Convert the last found forecast to a probability (0-1 scale)
            forecast = float(forecasts[-1]) / 100
            return forecast
        else:
            return None

    # Create mapping of prompt patterns to prompt IDs
    # This helps with accurate identification of prompt types
    prompt_patterns = {}
    for prompt_id in range(1, 39):  # Based on your documented 38 prompts
        prompt_rows = model_questions[model_questions["prompt_id"] == prompt_id]
        if not prompt_rows.empty:
            # Get a sample prompt text for this ID
            prompt_text = prompt_rows["prompt"].iloc[0]
            # Store the first 100 characters as a pattern
            prompt_patterns[prompt_id] = prompt_text[:100]

    def match_prompt_id(text):
        if not isinstance(text, str):
            return None

        best_match = None
        best_match_length = 0

        for prompt_id, pattern in prompt_patterns.items():
            if pattern in text:
                # If we find a longer matching pattern, use that
                if len(pattern) > best_match_length:
                    best_match = prompt_id
                    best_match_length = len(pattern)

        return best_match

    # Extract question IDs from text
    def extract_question_id(text):
        if not isinstance(text, str):
            return None

        match = re.search(r"meteofrance_TEMPERATURE_celsius\.\d+\.D", text)
        if match:
            return f"dbnomics_{match.group(0)}"
        return None

    # 3. Process sonnet responses
    print("Processing sonnet responses...")

    # Extract data from responses
    sonnet["forecast"] = sonnet["modelOutput_content_text"].apply(extract_forecast)
    sonnet["question_id"] = sonnet["modelInput_messages_content_text"].apply(
        extract_question_id
    )
    sonnet["prompt_id"] = sonnet["modelInput_messages_content_text"].apply(
        match_prompt_id
    )

    # Print extraction stats
    print(
        f"Successfully extracted {sonnet['forecast'].notna().sum()} forecasts out of {len(sonnet)} responses"
    )
    print(f"Successfully identified {sonnet['question_id'].notna().sum()} question IDs")
    print(f"Successfully matched {sonnet['prompt_id'].notna().sum()} prompt IDs")

    # 4. Merge with question outcomes
    print("Merging with question outcomes...")

    # Filter out responses without forecast, question_id, or prompt_id
    valid_responses = sonnet.dropna(subset=["forecast", "question_id", "prompt_id"])
    print(
        f"Valid responses with forecast, question ID, and prompt ID: {len(valid_responses)}"
    )

    # Merge with question outcomes
    responses_with_outcomes = pd.merge(
        valid_responses,
        questions_sample[["question_id", "resolved_to"]],
        on="question_id",
        how="inner",
    )

    print(
        f"Responses successfully merged with question outcomes: {len(responses_with_outcomes)}"
    )

    # 5. Calculate Brier scores
    print("Calculating Brier scores...")
    responses_with_outcomes["brier_score"] = (
        responses_with_outcomes["forecast"] - responses_with_outcomes["resolved_to"]
    ) ** 2

    # 6. Analyze performance by prompt type
    print("Analyzing performance by prompt type...")

    # Create prompt type names from the study documentation
    prompt_names = {
        1: "Control",
        2: "Chain of Thought",
        3: "Self-Consistency",
        4: "Few-Shot",
        5: "Step-Back",
        6: "High Personal Stakes",
        7: "Echo",
        8: "Structure",
        9: "Emotional Prompt",
        10: "Re-Reading",
        11: "Uncertainty Quantification",
        12: "Superforecasting Persona",
        13: "Abstention",
        14: "Counterfactual Reasoning",
        15: "Analogical Reasoning",
        16: "Hypothetical Scenario Analysis",
        17: "Scoring Rule",
        18: "Premortem",
        19: "Base Rate First",
        20: "Time Decomposition",
        21: "Metacognition",
        22: "Anti-Biasing (Anchoring)",
        23: "Anti-Biasing (Round Numbers)",
        24: "Anti-Biasing (Overconfidence)",
        25: "Frequency-Based Reasoning",
        26: "Propose-Evaluate-Select",
        27: "Bayesian reasoning",
        28: "Multiple reference classes",
        29: "Fermi estimate",
        30: "Self-critique",
        31: "Tipping",
        32: "Question paraphrasing",
        33: "Simulated dialogue",
        34: "Simulated debate",
        35: "Pros & Cons",
        36: "Event decomposition",
        37: "Deep breath",
        38: "Explicit uncertainty sources",
    }

    # Add prompt names to the dataframe
    responses_with_outcomes["prompt_name"] = responses_with_outcomes["prompt_id"].map(
        prompt_names
    )

    # Aggregate performance by prompt type
    prompt_performance = (
        responses_with_outcomes.groupby(["prompt_id", "prompt_name"])
        .agg({"brier_score": ["mean", "std", "count"], "forecast": ["mean", "std"]})
        .reset_index()
    )

    # Flatten the multi-level columns
    prompt_performance.columns = [
        "_".join(col).strip("_") for col in prompt_performance.columns.values
    ]

    # Sort by Brier score (lower is better)
    prompt_performance = prompt_performance.sort_values("brier_score_mean")

    # Calculate overall average Brier score
    overall_brier = responses_with_outcomes["brier_score"].mean()
    print(f"Overall average Brier score: {overall_brier:.4f}")

    # 7. Save detailed results
    print("Saving detailed results...")
    responses_with_outcomes.to_csv("sonnet_forecasts_with_brier.csv", index=False)
    prompt_performance.to_csv("sonnet_prompt_performance.csv", index=False)

    # 8. Generate summary report
    print("\nPrompt Performance Summary (lower Brier score is better):")
    summary_table = prompt_performance[
        [
            "prompt_id",
            "prompt_name",
            "brier_score_mean",
            "brier_score_std",
            "brier_score_count",
        ]
    ]
    summary_table.columns = [
        "Prompt ID",
        "Prompt Name",
        "Mean Brier Score",
        "Std Dev",
        "Count",
    ]
    print(summary_table.to_string(index=False))

    # 9. Analyze variance in performance across questions
    question_prompt_performance = responses_with_outcomes.pivot_table(
        index="prompt_id", columns="question_id", values="brier_score", aggfunc="mean"
    )

    prompt_consistency = question_prompt_performance.std(axis=1).sort_values()
    print("\nPrompt Consistency Across Questions (lower std dev is more consistent):")
    consistency_table = pd.DataFrame(
        {
            "Prompt ID": prompt_consistency.index,
            "Std Dev Across Questions": prompt_consistency.values,
        }
    )
    consistency_table["Prompt Name"] = consistency_table["Prompt ID"].map(prompt_names)
    print(consistency_table.to_string(index=False))

    # 10. Plot performance
    print("\nGenerating visualization...")
    try:
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=prompt_performance.sort_values("brier_score_mean").head(
                15
            ),  # Top 15 prompts
            x="brier_score_mean",
            y="prompt_name",
            palette="viridis",
        )
        plt.xlabel("Mean Brier Score (lower is better)")
        plt.ylabel("Prompt Type")
        plt.title("Top 15 Prompts by Forecast Accuracy (Brier Score)")
        plt.tight_layout()
        plt.savefig("prompt_performance.png")
        print("Visualization saved as 'prompt_performance.png'")
    except Exception as e:
        print(f"Error generating visualization: {e}")

    # Return the processed data for further analysis
    return responses_with_outcomes, prompt_performance


def compare_with_baseline():
    """
    Compare each prompt's performance with the control prompt (baseline)
    """
    # Load results
    results = pd.read_csv("sonnet_prompt_performance.csv")

    # Get control prompt score
    control_score = results[results["prompt_id"] == 1]["brier_score_mean"].values[0]

    # Calculate improvement over baseline
    results["improvement"] = control_score - results["brier_score_mean"]
    results["percent_improvement"] = (results["improvement"] / control_score) * 100

    # Sort by improvement
    results_sorted = results.sort_values("percent_improvement", ascending=False)

    print("\nPrompt Performance Compared to Baseline (Control Prompt):")
    comparison_table = results_sorted[
        [
            "prompt_id",
            "prompt_name",
            "brier_score_mean",
            "improvement",
            "percent_improvement",
        ]
    ]
    comparison_table.columns = [
        "Prompt ID",
        "Prompt Name",
        "Brier Score",
        "Improvement",
        "% Improvement",
    ]
    print(comparison_table.to_string(index=False))

    # Save results
    results_sorted.to_csv("prompt_improvement_over_baseline.csv", index=False)

    return results_sorted


if __name__ == "__main__":
    results, performance = analyze_forecast_accuracy()
    improvement_analysis = compare_with_baseline()

    print("\nAnalysis complete. Files saved:")
    print(
        "1. sonnet_forecasts_with_brier.csv - Detailed response-level data with Brier scores"
    )
    print("2. sonnet_prompt_performance.csv - Aggregated performance by prompt type")
    print(
        "3. prompt_improvement_over_baseline.csv - Comparison with baseline control prompt"
    )
    print("4. prompt_performance.png - Visualization of top-performing prompts")
