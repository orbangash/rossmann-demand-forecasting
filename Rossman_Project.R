# =========================================================
# Rossmann Store Sales - Demand Forecasting
# =========================================================

# ---------------------------------------------------------
# Section 1: Libraries and Data Loading
# ---------------------------------------------------------

# Install packages if not already installed
if(!require(tidyverse))    install.packages("tidyverse")
if(!require(caret))        install.packages("caret")
if(!require(lubridate))    install.packages("lubridate")
if(!require(scales))       install.packages("scales")
if(!require(knitr))        install.packages("knitr")
if(!require(rpart))        install.packages("rpart")
if(!require(rpart.plot))   install.packages("rpart.plot")
if(!require(randomForest)) install.packages("randomForest")

# Load libraries
library(tidyverse)
library(caret)
library(lubridate)
library(scales)
library(knitr)
library(rpart)
library(rpart.plot)
library(randomForest)


# ---------------------------------------------------------
# Load Data
# Unzip data if not already done
# Source: https://www.kaggle.com/c/rossmann-store-sales/data
# ---------------------------------------------------------

if(!file.exists("Data/train.csv")) {
  unzip("Data.zip", exdir = "Data")
}

train <- read.csv("Data/train.csv", stringsAsFactors = FALSE)
store <- read.csv("Data/store.csv", stringsAsFactors = FALSE)

# Merge store metadata into daily sales records
rossmann <- train %>%
  left_join(store, by = "Store")

# Convert Date from string to proper Date format
rossmann <- rossmann %>%
  mutate(Date = as.Date(Date))

# ---------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------
cat("Rows:"    , nrow(rossmann), "\n")
cat("Columns:" , ncol(rossmann), "\n")
cat("Date range:", as.character(min(rossmann$Date)),
    "to", as.character(max(rossmann$Date)), "\n")
# ---------------------------------------------------------
# Section 2: Data Familiarisation
# ---------------------------------------------------------

# Structure and column types
glimpse(rossmann)

# Missing values per column
missing_values <- colSums(is.na(rossmann))
print(missing_values[missing_values > 0])

# ---------------------------------------------------------
# Section 3: Exploratory Data Analysis
# ---------------------------------------------------------

# --- 3.1 Sales Distribution ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  ggplot(aes(x = Sales)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "black") +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  ggtitle("Distribution of Daily Sales (Open Stores Only)") +
  xlab("Daily Sales (Euros)") +
  ylab("Count")

# --- 3.2 Effect of Promotions ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  group_by(Promo) %>%
  summarize(
    Avg_Sales    = round(mean(Sales), 0),
    Median_Sales = round(median(Sales), 0),
    Count        = n()
  ) %>%
  mutate(Promo = ifelse(Promo == 1, "Promotion Active", 
                        "No Promotion")) %>%
  kable(caption = "Sales With vs Without Promotion")

# --- 3.3 Day of Week Pattern ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  group_by(DayOfWeek) %>%
  summarize(avg_sales = mean(Sales)) %>%
  ggplot(aes(x = DayOfWeek, y = avg_sales)) +
  geom_col(fill = "darkorange", color = "black") +
  scale_y_continuous(labels = scales::comma) +
  scale_x_continuous(breaks = 1:7,
                     labels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")) +
  ggtitle("Average Sales by Day of Week") +
  xlab("Day of Week") + ylab("Average Sales (Euros)")

# --- 3.4 Monthly Seasonality ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  mutate(Month = month(Date, label = TRUE)) %>%
  group_by(Month) %>%
  summarize(avg_sales = mean(Sales)) %>%
  ggplot(aes(x = Month, y = avg_sales)) +
  geom_col(fill = "seagreen", color = "black") +
  scale_y_continuous(labels = scales::comma) +
  ggtitle("Average Sales by Month") +
  xlab("Month") + ylab("Average Sales (Euros)")

# --- 3.5 Sales Trend Over Time ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  group_by(Date) %>%
  summarize(avg_sales = mean(Sales)) %>%
  ggplot(aes(x = Date, y = avg_sales)) +
  geom_line(color = "darkblue", alpha = 0.6) +
  geom_smooth(method = "loess", color = "red", span = 0.1) +
  scale_y_continuous(labels = scales::comma) +
  ggtitle("Average Daily Sales Over Time") +
  xlab("Date") + ylab("Average Sales (Euros)")

# --- 3.6 Store Type Comparison ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  group_by(StoreType) %>%
  summarize(
    Avg_Sales = round(mean(Sales), 0),
    Count     = n()
  ) %>%
  kable(caption = "Average Sales by Store Type")

# --- 3.7 Competitor Distance Effect ---
rossmann %>%
  filter(Open == 1, Sales > 0, !is.na(CompetitionDistance)) %>%
  mutate(dist_bucket = cut(CompetitionDistance,
                           breaks = c(0, 500, 1000, 2000, 5000, 10000, Inf),
                           labels = c("<500m","500m-1km","1-2km",
                                      "2-5km","5-10km",">10km"))) %>%
  group_by(dist_bucket) %>%
  summarize(avg_sales = mean(Sales)) %>%
  ggplot(aes(x = dist_bucket, y = avg_sales)) +
  geom_col(fill = "purple", color = "black") +
  scale_y_continuous(labels = scales::comma) +
  ggtitle("Average Sales by Competitor Distance") +
  xlab("Distance to Nearest Competitor") +
  ylab("Average Sales (Euros)")

# --- 3.8 Competitor Distance - Sales per Customer ---
rossmann %>%
  filter(Open == 1, Sales > 0, Customers > 0,
         !is.na(CompetitionDistance)) %>%
  mutate(
    sales_per_customer = Sales / Customers,
    dist_bucket = cut(CompetitionDistance,
                      breaks = c(0, 500, 1000, 2000, 5000, 10000, Inf),
                      labels = c("<500m","500m-1km","1-2km",
                                 "2-5km","5-10km",">10km"))) %>%
  group_by(dist_bucket) %>%
  summarize(
    avg_sales_per_customer = mean(sales_per_customer),
    avg_customers          = mean(Customers),
    avg_sales              = mean(Sales)
  ) %>%
  kable(caption = "Sales Per Customer by Competitor Distance")

# --- 3.9 State Holiday Effect ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  mutate(StateHoliday = case_when(
    StateHoliday == "a" ~ "Public Holiday",
    StateHoliday == "b" ~ "Easter Holiday",
    StateHoliday == "c" ~ "Christmas",
    TRUE                ~ "No Holiday"
  )) %>%
  group_by(StateHoliday) %>%
  summarize(
    Avg_Sales    = round(mean(Sales), 0),
    Median_Sales = round(median(Sales), 0),
    Count        = n()
  ) %>%
  kable(caption = "Sales by State Holiday Type")

# --- 3.10 School Holiday Effect ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  group_by(SchoolHoliday) %>%
  summarize(
    Avg_Sales    = round(mean(Sales), 0),
    Median_Sales = round(median(Sales), 0),
    Count        = n()
  ) %>%
  mutate(SchoolHoliday = ifelse(SchoolHoliday == 1,
                                "School Holiday",
                                "Normal Day")) %>%
  kable(caption = "Sales: School Holiday vs Normal Day")


# --- 3.11 Assortment Effect ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  mutate(Assortment = case_when(
    Assortment == "a" ~ "Basic",
    Assortment == "b" ~ "Extra",
    Assortment == "c" ~ "Extended"
  )) %>%
  group_by(Assortment) %>%
  summarize(
    Avg_Sales    = round(mean(Sales), 0),
    Median_Sales = round(median(Sales), 0),
    Count        = n()
  ) %>%
  kable(caption = "Sales by Assortment Level")

rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  mutate(Assortment = case_when(
    Assortment == "a" ~ "Basic",
    Assortment == "b" ~ "Extra",
    Assortment == "c" ~ "Extended"
  )) %>%
  ggplot(aes(x = Assortment, y = Sales, fill = Assortment)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma) +
  ggtitle("Sales Distribution by Assortment Level") +
  xlab("Assortment Level") + ylab("Daily Sales (Euros)") +
  theme(legend.position = "none")

# --- 3.12 Promo2 Effect ---
rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  group_by(Promo, Promo2) %>%
  summarize(
    Avg_Sales = round(mean(Sales), 0),
    Count     = n(),
    .groups   = "drop"
  ) %>%
  mutate(
    Promo  = ifelse(Promo  == 1, "Promo Active",  "No Promo"),
    Promo2 = ifelse(Promo2 == 1, "Promo2 Active", "No Promo2")
  ) %>%
  kable(caption = "Sales by Promo and Promo2 Combination")

rossmann %>%
  filter(Open == 1, Sales > 0) %>%
  mutate(
    Promo  = ifelse(Promo  == 1, "Promo Active",  "No Promo"),
    Promo2 = ifelse(Promo2 == 1, "Promo2 Active", "No Promo2")
  ) %>%
  ggplot(aes(x = Promo, y = Sales, fill = Promo2)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma) +
  ggtitle("Sales by Promo and Promo2 Combination") +
  xlab("") + ylab("Daily Sales (Euros)") +
  labs(fill = "Promo2 Status")

# ---------------------------------------------------------
# Section 4: Feature Engineering
# ---------------------------------------------------------

# --- 4.1 Filter, Extract and Select ---
rossmann_model <- rossmann %>%
  
  # Remove closed stores - zero sales not a demand signal
  filter(Open == 1, Sales > 0) %>%
  
  # Extract date features
  mutate(
    Month = month(Date),
    Year  = year(Date)
  ) %>%
  
  # Impute missing CompetitionDistance with median
  mutate(CompetitionDistance = ifelse(
    is.na(CompetitionDistance),
    median(CompetitionDistance, na.rm = TRUE),
    CompetitionDistance
  )) %>%
  
  # Convert categoricals to factors
  mutate(
    Store        = as.factor(Store),
    StoreType    = as.factor(StoreType),
    Assortment   = as.factor(Assortment),
    DayOfWeek    = as.factor(DayOfWeek),
    Promo        = as.factor(Promo),
    Month        = as.factor(Month),
    Year         = as.factor(Year)
  ) %>%
  
  # Keep only confirmed features and target
  select(Sales, Store, DayOfWeek, Promo,
         StoreType, Assortment,
         CompetitionDistance, Month, Year, Date)

# --- 4.2 Verify ---
glimpse(rossmann_model)
cat("Rows after filtering:", nrow(rossmann_model), "\n")
cat("Any missing values:", anyNA(rossmann_model), "\n")

# ---------------------------------------------------------
# Section 5: Modelling
# ---------------------------------------------------------

# --- 5.1 Train/Test Split (Time-Based) ---
split_date <- as.Date("2015-06-19")  # Last 6 weeks of data

train_set <- rossmann_model %>%
  filter(Date < split_date) %>%
  select(-Date)

test_set <- rossmann_model %>%
  filter(Date >= split_date) %>%
  select(-Date)

# Verify split
cat("Training rows:", nrow(train_set), "\n")
cat("Test rows:"    , nrow(test_set),  "\n")
cat("Train date range: up to", as.character(split_date), "\n")
cat("Test date range: from", as.character(split_date),
    "to 2015-07-31\n")

# Define RMSE function
RMSE <- function(true, predicted){
  sqrt(mean((true - predicted)^2))
}

# --- 5.2 Baseline Model (Store Average) ---

# Calculate average sales per store from training set
store_avgs <- train_set %>%
  group_by(Store) %>%
  summarize(pred = mean(Sales))

# Predict on test set
baseline_pred <- test_set %>%
  left_join(store_avgs, by = "Store") %>%
  pull(pred)

# Calculate RMSE
rmse_baseline <- RMSE(test_set$Sales, baseline_pred)
cat("Baseline RMSE:", rmse_baseline, "\n")

# Start results tracking table
rmse_results <- tibble(
  Model = "Baseline (Store Average)",
  RMSE  = rmse_baseline
)
kable(rmse_results, caption = "Model Performance Tracking")

# --- 5.3 Linear Regression ---

# Train linear regression model
lm_model <- lm(Sales ~ ., data = train_set)

# Predict on test set
lm_pred <- predict(lm_model, newdata = test_set)

# Clip negative predictions to zero
# (sales can never be negative)
lm_pred <- pmax(lm_pred, 0)

# Calculate RMSE
rmse_lm <- RMSE(test_set$Sales, lm_pred)
cat("Linear Regression RMSE:", rmse_lm, "\n")

# Add to results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Linear Regression", RMSE = rmse_lm))
kable(rmse_results, caption = "Model Performance Tracking")

# --- 5.4 Decision Tree ---
tree_model <- rpart(Sales ~ ., 
                    data       = train_set,
                    method     = "anova",
                    control    = rpart.control(maxdepth = 10))

# Visualise the tree
rpart.plot(tree_model, 
           main   = "Decision Tree Structure",
           type   = 4,
           extra  = 101)

# Predict on test set
tree_pred <- predict(tree_model, newdata = test_set)
tree_pred <- pmax(tree_pred, 0)

# Calculate RMSE
rmse_tree <- RMSE(test_set$Sales, tree_pred)
cat("Decision Tree RMSE:", rmse_tree, "\n")

# Add to results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Decision Tree", RMSE = rmse_tree))
kable(rmse_results, caption = "Model Performance Tracking")

# --- 5.5 Random Forest ---
# randomforest cannot handle factors with more than 53 categories
# Convert Store to numeric to resolve this

set.seed(42)
train_rf <- train_set %>%
  mutate(Store = as.numeric(as.character(Store))) %>%
  sample_n(50000)  # Random sample of 50K rows

test_rf <- test_set %>%
  mutate(Store = as.numeric(as.character(Store)))

rf_model <- randomForest(
  Sales ~ .,
  data       = train_rf,
  ntree      = 100,
  mtry       = 3,
  nodesize   = 25,
  importance = TRUE
)

rf_pred <- predict(rf_model, newdata = test_rf)
rf_pred <- pmax(rf_pred, 0)

rmse_rf <- RMSE(test_set$Sales, rf_pred)
cat("Random Forest RMSE:", rmse_rf, "\n")

rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Random Forest", RMSE = rmse_rf))
kable(rmse_results, caption = "Model Performance Tracking")

# --- 5.6 Feature Importance ---
importance_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "%IncMSE"]
) %>%
  arrange(desc(Importance))

ggplot(importance_df, 
       aes(x = reorder(Feature, Importance), 
           y = Importance)) +
  geom_col(fill = "steelblue", color = "black") +
  coord_flip() +
  ggtitle("Random Forest - Feature Importance") +
  xlab("Feature") +
  ylab("% Increase in MSE if Removed")

# ---------------------------------------------------------
# Section 6: Final Results
# ---------------------------------------------------------

# Display final model comparison
cat("\n=== FINAL MODEL COMPARISON ===\n\n")
kable(rmse_results, caption = "Final Model Performance Summary")

# Identify best model
best_model <- rmse_results %>% slice_min(RMSE, n = 1)
cat("\nBest Model:", best_model$Model, "\n")
cat("Best RMSE:", round(best_model$RMSE, 2), "\n")
cat("Improvement over baseline:", 
    round(rmse_baseline - best_model$RMSE, 2), " (",
    round((1 - best_model$RMSE / rmse_baseline) * 100, 1), "% reduction)\n")