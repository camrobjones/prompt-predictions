// Simulate data and test multiple comparisons
library(tidyverse)
library(lme4)
library(lmerTest)

# Set parameters
n_questions <- 50
n_prompts <- 49  # treatment prompts
n_sims <- 50
effect_sizes <- seq(0.01, 0.1, by=0.01)  # range of effect sizes to test

# Function to run one simulation
run_sim <- function(effect_size) {
  # Generate data
  data <- expand.grid(
    question_id = 1:n_questions,
    prompt_id = 0:(n_prompts)  # 0 is control
  ) %>%
    mutate(
      # Random effect for questions
      question_effect = rnorm(n_questions)[question_id],
      # Treatment effect (0 for control, effect_size for treatment)
      treatment_effect = ifelse(prompt_id == 0, 0, effect_size),
      # Generate Brier scores (adding some noise)
      brier_score = 0.3 + question_effect + treatment_effect + rnorm(n(), sd=0.1)
    )
  
  # Fit mixed effects model
  model <- lmer(brier_score ~ factor(prompt_id) + (1|question_id), data=data)
  
  # Extract p-values for treatment effects
  sum <- summary(model)
  p_vals <- sum$coefficients[-1, "Pr(>|t|)"]
  
  # Apply B-H correction
  p_adj <- p.adjust(p_vals, method="BH")
  
  # Calculate power (proportion of significant results)
  power <- mean(p_adj < 0.05)
  
  return(power)
}

# Run simulations for each effect size
results <- tibble(
  effect_size = effect_sizes,
  power = map_dbl(effect_sizes, ~mean(replicate(n_sims/10, run_sim(.))))
)

print("Power by effect size:")
print(results)

# Plot results
ggplot(results, aes(x=effect_size, y=power)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels=scales::percent) +
  labs(
    x="Effect Size (Difference in Brier Score)",
    y="Power (% of Effects Detected)",
    title="Power Analysis for Multiple Comparisons"
  )
