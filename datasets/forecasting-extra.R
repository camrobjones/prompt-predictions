
## 10. Extend Analysis to Other Models

Now let's extend this analysis to the other models:

```{r other_models, eval=FALSE}
# Function to process model data
process_model <- function(model_name, textcol) {
  # Load data
  model_data <- read_csv(paste0("data/", model_name, ".csv"))
  
  cat("\nProcessing", model_name, "model with", nrow(model_data), "responses\n")
  
  # Extract forecasts and match with model_questions
  model_processed <- model_data %>%
    mutate(
      forecast = map_dbl(modelOutput_content_text, extract_forecast, .default = NA),
      record_num = as.numeric(str_extract(recordId, "\\d+")) - 1
    ) %>%
    left_join(
      model_questions %>% select(id, question_id, prompt_id),
      by = c("record_num" = "id")
    ) %>%
    mutate(model = model_name) %>%
    select(
      model, recordId, question_id, prompt_id, forecast,
      input_text = modelInput_messages_content_text,
      output_text = modelOutput_content_text
    )
  
  # Merge with outcomes and calculate Brier scores
  model_with_brier <- model_processed %>%
    inner_join(
      questions_sample %>% select(question_id, resolved_to),
      by = "question_id"
    ) %>%
    mutate(brier_score = (forecast - resolved_to)^2)
  
  # Report statistics
  valid_rows <- nrow(model_with_brier)
  mean_brier <- mean(model_with_brier$brier_score, na.rm = TRUE)
  
  cat("Valid rows with Brier scores:", valid_rows, "\n")
  cat("Overall Brier score:", round(mean_brier, 4), "\n")
  
  return(model_with_brier)
}

# Process all models
model_list <- c("gpt4o", "haiku", "llama")
text_cols <- c("choice_message_content",)
all_models_data <- list()

for (model in model_list) {
  if (file.exists(paste0("data/", model, ".csv"))) {
    all_models_data[[model]] <- process_model(model)
  } else {
    cat("Data file for", model, "not found, skipping...\n")
  }
}

# Add sonnet data to the list
all_models_data[["sonnet"]] <- sonnet_with_brier

# Combine all model data
combined_data <- bind_rows(all_models_data)

# Check distribution of data by model
model_counts <- combined_data %>%
  group_by(model) %>%
  summarize(
    count = n(),
    mean_brier = mean(brier_score, na.rm = TRUE),
    .groups = "drop"
  )

model_counts %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), font_size = 11)
```

## 11. Cross-Model Analysis

Now let's compare performance across models:
  
  ```{r cross_model, eval=FALSE}
# Add prompt names to combined data
combined_data_with_names <- combined_data %>%
  left_join(prompt_names, by = "prompt_id")

# Calculate prompt performance by model
prompt_by_model <- combined_data_with_names %>%
  group_by(model, prompt_id, prompt_name) %>%
  summarize(
    mean_brier = mean(brier_score, na.rm = TRUE),
    sd_brier = sd(brier_score, na.rm = TRUE),
    count = n(),
    .groups = "drop"
  )

# Get control scores for each model
control_by_model <- prompt_by_model %>%
  filter(prompt_id == 1) %>%
  select(model, control_score = mean_brier)

# Calculate improvement over control for each model
prompt_improvement <- prompt_by_model %>%
  left_join(control_by_model, by = "model") %>%
  mutate(
    improvement = control_score - mean_brier,
    percent_improvement = (improvement / control_score) * 100
  )

# Show top 5 prompts for each model
top_prompts_by_model <- prompt_improvement %>%
  group_by(model) %>%
  top_n(5, percent_improvement) %>%
  arrange(model, desc(percent_improvement))

top_prompts_by_model %>%
  select(model, prompt_id, prompt_name, mean_brier, percent_improvement, count) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), font_size = 11)

# Create a wide format table of Brier scores by prompt and model
prompt_performance_wide <- prompt_by_model %>%
  filter(count >= 10) %>%  # Only include prompts with sufficient data
  select(model, prompt_id, prompt_name, mean_brier) %>%
  pivot_wider(
    names_from = model,
    values_from = mean_brier
  ) %>%
  rowwise() %>%
  mutate(
    available_models = sum(!is.na(c_across(where(is.numeric)))),
    average = mean(c_across(where(is.numeric)), na.rm = TRUE),
    std_dev = sd(c_across(where(is.numeric)), na.rm = TRUE)
  ) %>%
  arrange(average) %>%
  ungroup()

# Show top 10 prompts by average performance
prompt_performance_wide %>%
  filter(available_models >= 2) %>%  # Only include prompts with data from at least 2 models
  top_n(10, -average) %>%
  select(prompt_id, prompt_name, average, std_dev, everything()) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), font_size = 11)
```

## 12. Visualize Cross-Model Comparisons

Let's create visualizations of cross-model performance:

```{r cross_model_viz, eval=FALSE}
# Overall model comparison
model_counts %>%
  ggplot(aes(x = reorder(model, mean_brier), y = mean_brier, fill = model)) +
  geom_col() +
  geom_text(aes(label = round(mean_brier, 4)), vjust = -0.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Overall Model Performance Comparison",
    x = "Model",
    y = "Mean Brier Score (lower is better)",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# Prompt performance across models (top 5 by average)
top_prompts <- prompt_performance_wide %>%
  filter(available_models >= 2) %>%
  top_n(5, -average) %>%
  pull(prompt_id)

# Include control prompt for comparison
comparison_prompts <- c(1, top_prompts)

# Create a long format dataset for visualization
model_columns <- intersect(names(prompt_performance_wide), c("sonnet", "gpt4o", "haiku", "llama"))

prompt_comparison_long <- prompt_performance_wide %>%
  filter(prompt_id %in% comparison_prompts) %>%
  select(prompt_id, prompt_name, all_of(model_columns)) %>%
  pivot_longer(
    cols = all_of(model_columns),
    names_to = "model",
    values_to = "brier_score"
  )

# Create visualization
prompt_comparison_long %>%
  mutate(
    prompt_name = factor(prompt_name, 
                       levels = c("Control", prompt_names$prompt_name[prompt_names$prompt_id %in% top_prompts])),
    model = factor(model)
  ) %>%
  ggplot(aes(x = prompt_name, y = brier_score, fill = model)) +
  geom_col(position = "dodge") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Top 5 Prompts Performance Across Models",
    subtitle = "Compared to Control Prompt",
    x = "Prompt Type",
    y = "Mean Brier Score (lower is better)",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation of prompt effects between models
# Only include prompts with data from at least 2 models
correlation_data <- prompt_performance_wide %>%
  filter(available_models >= 2) %>%
  select(all_of(model_columns))

# Calculate correlation matrix
if (ncol(correlation_data) >= 2) {
  prompt_correlation <- cor(correlation_data, use = "pairwise.complete.obs")

  # Visualization of correlation matrix
  prompt_correlation %>%
    as.data.frame() %>%
    rownames_to_column("model1") %>%
    pivot_longer(
      cols = -model1,
      names_to = "model2",
      values_to = "correlation"
    ) %>%
    ggplot(aes(x = model1, y = model2, fill = correlation)) +
    geom_tile() +
    geom_text(aes(label = round(correlation, 2)), color = "white") +
    scale_fill_viridis() +
    labs(
      title = "Correlation of Prompt Effects Between Models",
      x = "Model",
      y = "Model",
      fill = "Correlation"
    ) +
    theme_minimal()
}
```

## 13. Most Universal Prompts

Finally, let's identify the prompts that work well across all models:
  
  ```{r universal_prompts, eval=FALSE}
# Calculate ranks within each model
prompt_ranks <- prompt_by_model %>%
  filter(count >= 10) %>%  # Only include prompts with sufficient data
  group_by(model) %>%
  mutate(rank = rank(mean_brier)) %>%
  group_by(prompt_id, prompt_name) %>%
  summarize(
    model_count = n_distinct(model),
    avg_rank = mean(rank, na.rm = TRUE),
    std_rank = sd(rank, na.rm = TRUE),
    avg_brier = mean(mean_brier, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(avg_rank)

# Show top universal prompts
prompt_ranks %>%
  filter(model_count >= 2) %>%  # Only include prompts with data from at least 2 models
  top_n(10, -avg_rank) %>%
  select(prompt_id, prompt_name, model_count, avg_rank, std_rank, avg_brier) %>%
  kable(caption = "Most universal prompts across models (by average rank)") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), font_size = 11)

# Show most consistent prompts (lowest standard deviation in rank)
prompt_ranks %>%
  filter(model_count >= 2) %>%  # Only include prompts with data from at least 2 models
  top_n(10, -std_rank) %>%
  arrange(std_rank) %>%
  select(prompt_id, prompt_name, model_count, avg_rank, std_rank, avg_brier) %>%
  kable(caption = "Most consistent prompts across models (by rank stability)") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), font_size = 11)

# Visualize universal prompts
prompt_ranks %>%
  filter(model_count >= 2) %>%  # Only include prompts with data from at least 2 models
  top_n(10, -avg_rank) %>%
  mutate(prompt_name = reorder(prompt_name, -avg_rank)) %>%
  ggplot(aes(x = prompt_name, y = avg_rank, fill = avg_brier)) +
  geom_col() +
  scale_fill_viridis() +
  labs(
    title = "Most Universal Prompts Across All Models",
    subtitle = "Based on average rank (lower is better)",
    x = "Prompt Type",
    y = "Average Rank",
    fill = "Avg Brier Score"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

## 14. Conclusions

Based on our analysis, we can draw the following conclusions:
  
  1. The best performing prompts for Sonnet were: `r if(exists("prompt_performance")) {paste(prompt_performance %>% filter(count >= 10) %>% top_n(3, percent_improvement) %>% pull(prompt_name), collapse=", ")}`

2. The worst performing prompts were: `r if(exists("prompt_performance")) {paste(prompt_performance %>% filter(count >= 10) %>% top_n(3, -percent_improvement) %>% pull(prompt_name), collapse=", ")}`

3. The prompt with the most consistent performance was: `r if(exists("prompt_consistency")) {paste(prompt_consistency %>% filter(question_count >= 5) %>% arrange(question_std) %>% slice(1) %>% pull(prompt_name))}`

4. Statistical significance: `r if(exists("t_test_result")) {if(t_test_result$p.value < 0.05) {"The difference between the best prompt and control is statistically significant."} else {"The difference between the best prompt and control is not statistically significant."}}`

5. `r if(exists("prompt_ranks")) {paste0("Across all models, the most universal prompt was: ", prompt_ranks %>% slice(1) %>% pull(prompt_name))}`

6. Sample imbalance: The Control prompt has significantly more samples than other prompts, which could affect the reliability of our comparisons.

This analysis provides insights into how different prompting strategies affect the forecasting accuracy of large language models, with important implications for building more accurate forecasting systems.
