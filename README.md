# Prompt Predictions

Testing whether different prompt strategies improve LLM forecasting performance.

## Setup

1. Clone this repo
2. Clone [forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets/tree/main) into an adjacent directory named `forecasting-datasets`

## Files

- `convert_json.py`: Converts ForecastBench JSON data to CSV format
- `sample_questions.py`: Samples questions across different data sources (ACLED, Wikipedia, FRED, etc.)
- `power_analysis.R`: Statistical power analysis for experiment design

## Usage

1. Run `convert_json.py` to generate merged dataset
2. Use `sample_questions.py` to extract balanced samples for testing

Dataset params need to be edited inside the files.