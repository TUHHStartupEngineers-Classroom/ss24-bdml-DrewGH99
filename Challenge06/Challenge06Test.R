library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)
library(lime)
library(rsample)
library(ggplot2)

# h2o modeling ----
employee_attrition_tbl <- read_csv("Challenge 4/datasets-1067-1925-WA_Fn-UseC_-HR-Employee-Attrition.csv")
definitions_raw_tbl    <- read_excel("Challenge 4/data_definitions.xlsx", sheet = 1, col_names = FALSE)

# Processing Pipeline
source("Challenge 4/data_processing_pipeline.R")

employee_attrition_readable_tbl <- process_hr_data_readable(employee_attrition_tbl, definitions_raw_tbl)

# Split into test and train
set.seed(seed = 1113)
split_obj <- rsample::initial_split(employee_attrition_readable_tbl, prop = 0.85)

# Assign training and test data
train_readable_tbl <- training(split_obj)
test_readable_tbl  <- testing(split_obj)

# ML Preprocessing Recipe 
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
  step_zv(all_predictors()) %>%
  step_mutate_at(c("JobLevel", "StockOptionLevel"), fn = as.factor) %>% 
  prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_readable_tbl)

h2o.init()

## Split data ----
split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 1234)
train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_tbl)

## Set the target and predictors ----
y <- "Attrition"
x <- setdiff(names(train_h2o), y)

## train data

automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame    = train_h2o,
  validation_frame  = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs  = 30,
  nfolds            = 5 
)

automl_leader_id <- automl_models_h2o@leader@model_id
path <- ("h20_models/Challenge 6/")
automl_leader_path <- paste0(path, automl_leader_id)

h2o.getModel(automl_models_h2o@leader@model_id) %>%
  h2o.saveModel(path = "h20_models/Challenge 6/", force = T)

automl_leader <- h2o.loadModel(automl_leader_path)

predictions_tbl <- automl_leader %>% 
  h2o.predict(newdata = as.h2o(test_tbl)) %>%
  as.tibble() %>%
  bind_cols(
    test_tbl %>%
      select(Attrition, EmployeeNumber)
  )

test_tbl %>%
  slice(1) %>%
  glimpse()

# Create Explainer ----

explainer <- train_tbl %>%
  select(-Attrition) %>%
  lime(
    model           = automl_leader,
    bin_continuous  = TRUE,
    n_bins          = 4,
    quantile_bins   = TRUE
  )

# Create Explanation ----

explanation <- test_tbl %>%
  slice(1) %>%
  select(-Attrition) %>%
  lime::explain(
    
    # Pass our explainer object
    explainer = explainer,
    # Because it is a binary classification model: 1
    n_labels   = 1,
    # number of features to be returned
    n_features = 8,
    # number of localized linear models
    n_permutations = 5000,
    # Let's start with 1
    kernel_width   = 1
  )

explanation %>%
  as.tibble() %>%
  select(feature:prediction) 

g <- plot_features(explanation = explanation, ncol = 1)

## Multiple Explanations ----

explanation <- test_tbl %>%
  slice(1:20) %>%
  select(-Attrition) %>%
  lime::explain(
    explainer = explainer,
    n_labels   = 1,
    n_features = 8,
    n_permutations = 5000,
    kernel_width   = 0.5
  )

explanation %>%
  as.tibble()

plot_features(explanation, ncol = 4)

plot_explanations(explanation)


# Challenge Part 1 ----

explanation %>% 
  as.tibble()

case_1 <- explanation %>%
  filter(case == 1)

case_1 %>%
  plot_features()


label_both_upper <- function(labels, multi_line = TRUE, sep = ': ') {
  #names(labels) <- toTitleCase(names(labels))
  label_both(labels, multi_line, sep)
}


case_1$type <- factor(ifelse(sign(case_1$feature_weight) == 1, 'Supports', 'Contradicts'))
description <- paste0(case_1$case, '_', case_1[['label']])
desc_width <- max(nchar(description)) + 1
description <- paste0(format(description, width = desc_width), case_1$feature_desc)
case_1$description <- factor(description, levels = description[order(abs(case_1$feature_weight))])
case_1$case <- factor(case_1$case, unique(case_1$case))
case_1$`case_1 fit` <- format(case_1$model_r2, digits = 2)


case_1$probability <- format(case_1$label_prob, digits = 2)
case_1$label <- factor(case_1$label, unique(case_1$label[order(case_1$label_prob, decreasing = TRUE)]))
p <- ggplot(case_1) +
  facet_wrap(~ case + label + probability + `case_1 fit`, scales = 'free_y', ncol = 2, labeller = label_both_upper)

p +
  geom_col(aes_(~description, ~feature_weight, fill = ~type)) +
  coord_flip() +
  scale_fill_manual(values = c('lightcoral', 'green3'), drop = FALSE) +
  scale_x_discrete(labels = function(lab) substr(lab, desc_width + 1, nchar(lab))) +
  labs(y = 'Weight', x = 'Feature', fill = '') +
  theme_minimal()


# Challenge Part 2 ----

explanation$feature_desc <- factor(
  explanation$feature_desc,
  levels = rev(unique(explanation$feature_desc[order(explanation$feature, explanation$feature_value)]))
)
p <- ggplot(explanation, aes_(~case, ~feature_desc)) +
  geom_tile(aes_(fill = ~feature_weight)) +
  scale_x_discrete('Case', expand = c(0, 0)) +
  scale_y_discrete('Feature', expand = c(0, 0)) +
  scale_fill_gradient2('Feature\nweight', low = 'lightcoral', mid = '#f7f7f7', high = 'green3') +
  theme_minimal() +
  theme(panel.border = element_rect(fill = NA, colour = 'grey60', size = 1),
        panel.grid = element_blank(),
        legend.position = 'right',
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
p
