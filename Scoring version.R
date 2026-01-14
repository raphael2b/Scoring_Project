# Install packages

install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("moments")
install.packages("car")
install.packages("MASS")    # Note : MASS est inclus dans R, mais peut nécessiter une installation explicite dans certains cas
install.packages("pROC")
install.packages("corrplot")
install.packages("GGally")
install.packages("stargazer")

# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(stats)
library(moments)  # For skewness and kurtosis
library(car)      # For Levene's test
library(MASS)     # For logistic regression
library(pROC)     # For ROC curves
library(corrplot) # For correlation matrix
library(GGally)   # For pair plots
library(stargazer)# For LaTeX table export

# Suppress warnings (similar to Python's warnings.simplefilter)
options(warn = -1)

# Define variable labels (similar to Python's dictionary)
variables <- list(
  yd = "Financial Difficulty",
  tdta = "Debt/Assets",
  reta = "Retained Earnings",
  opita = "Income/Assets",
  ebita = "Pre-Tax Earnings/Assets",
  lsls = "Log Sales",
  lta = "Log Assets",
  gempl = "Employment Growth",
  invsls = "Inventory/Sales",
  nwcta = "Net Working Capital/Assets",
  cacl = "Current Assets/Liabilities",
  qacl = "Quick Assets/Liabilities",
  fata = "Fixed Assets/Total Assets",
  ltdta = "Long-Term Debt/Total Assets",
  mveltd = "Market Value Equity/Long-Term Debt"
)

# Load the dataset
df <- read.csv("C:/Users/jbchatelai/Downloads/defaut2000.csv", sep = ";")

# Replace commas with dots and convert to numeric
df <- df %>%
  mutate_all(~as.numeric(gsub(",", ".", .)))

# Check first 5 rows
head(df, 5)

# Dimensions of the dataset
dim(df)

# Variable types
str(df)

# Descriptive statistics for first eight variables
summary(df[, 1:8])

# Replace -99.99 with NA for 'fata' and 'ltdta'
df <- df %>%
  mutate(fata = na_if(fata, -99.99),
         ltdta = na_if(ltdta, -99.99))

# Check descriptive statistics for 'fata' and 'ltdta'
summary(df[, c("fata", "ltdta")])

# Sort by 'yd' and 'reta' and reset index
df_sorted <- df %>%
  arrange(yd, reta) %>%
  mutate(row_num = row_number())

# Check first 5 rows of sorted data
head(df_sorted[, c("yd", "reta", "tdta")], 5)

# Separate target (y) and features (X)
y_sorted <- df_sorted$yd
X_sorted <- dplyr::select(df_sorted, -yd)

# Verify results
print(y_sorted)
print(X_sorted)

# Split into train (odd rows) and test (even rows)
X_train <- X_sorted[seq(2, nrow(X_sorted), 2), ]
X_test <- X_sorted[seq(1, nrow(X_sorted), 1), ]
y_train <- y_sorted[seq(2, length(y_sorted), 2)]
y_test <- y_sorted[seq(1, length(y_sorted), 1)]

# Combine y_train and X_train for analysis
y_X_train <- bind_cols(yd = y_train, X_train)

# Separate train set into defaulting and non-defaulting groups
X_train_safe <- X_train[y_train == 0, ]
X_train_default <- X_train[y_train == 1, ]

# Descriptive statistics for 'reta' and 'tdta' in both groups
summary(X_train_safe[, c("reta", "tdta")])
summary(X_train_default[, c("reta", "tdta")])

# Bivariate analysis: Pair plot with KDE and regression lines
vars <- c("yd", "tdta", "reta")
new_df <- y_X_train %>% mutate(target = factor(yd, levels = c(0, 1), labels = c("Non-Default", "Default")))

# Check for zero variance
sapply(vars, function(var) var(y_X_train[[var]], na.rm = TRUE))

# Create pair plot
p <- ggpairs(new_df,
             columns = vars,
             aes(colour = target, fill = target),
             diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
             lower = list(continuous = wrap("points", alpha = 0.7) + 
                            geom_smooth(method = "lm", color = "blue", se = FALSE)),
             upper = list(continuous = "blank")) +
  scale_color_manual(values = c("green", "red")) +
  scale_fill_manual(values = c("green", "red")) +
  theme_minimal()
print(p)

# Univariate distributions and boxplots for 'tdta'
col <- "tdta"
groups <- split(y_X_train, y_X_train$yd)
x_min <- min(y_X_train[[col]], na.rm = TRUE)
x_max <- max(y_X_train[[col]], na.rm = TRUE)
x <- seq(x_min, x_max, length.out = 100)

# Calculate max density for consistent y-axis
max_density <- 0
group_stats <- list()
for (yd in names(groups)) {
  data <- groups[[yd]][[col]] %>% na.omit()
  dens <- density(data)
  max_density <- max(max_density, max(dens$y))
}
y_max <- max_density * 1.1

# Create subplots
p1 <- ggplot(y_X_train, aes_string(x = col, fill = factor(y_X_train$yd))) +
  geom_histogram(aes(y = ..density..), alpha = 0.4, bins = 30) +
  geom_density(alpha = 0.2) +
  stat_function(fun = dnorm, args = list(mean = mean(y_X_train[y_X_train$yd == 0, col], na.rm = TRUE),
                                         sd = sd(y_X_train[y_X_train$yd == 0, col], na.rm = TRUE)),
                color = "darkgreen", linetype = "dashed") +
  stat_function(fun = dnorm, args = list(mean = mean(y_X_train[y_X_train$yd == 1, col], na.rm = TRUE),
                                         sd = sd(y_X_train[y_X_train$yd == 1, col], na.rm = TRUE)),
                color = "darkred", linetype = "dashed") +
  facet_wrap(~yd, labeller = labeller(yd = c("0" = "Non-Default (yd=0)", "1" = "Default (yd=1)"))) +
  ylim(0, y_max) +
  theme_minimal() +
  labs(title = paste("Distribution of", col), y = "Density")

p2 <- ggplot(y_X_train, aes_string(x = col, fill = factor(y_X_train$yd))) +
  geom_boxplot() +
  facet_wrap(~yd, labeller = labeller(yd = c("0" = "Non-Default (yd=0)", "1" = "Default (yd=1)"))) +
  theme_minimal() +
  labs(title = paste("Boxplot of", col), x = col, y = "")

# Combine plots
gridExtra::grid.arrange(p1, p2, ncol = 1, heights = c(2, 1))

# Histogram and KDE for 'tdta' by group
p <- ggplot() +
  geom_histogram(data = filter(y_X_train, yd == 0), aes(x = tdta, y = ..density.., fill = "Non-Default"), alpha = 0.4) +
  geom_histogram(data = filter(y_X_train, yd == 1), aes(x = tdta, y = ..density.., fill = "Default"), alpha = 0.4) +
  geom_density(data = filter(y_X_train, yd == 0), aes(x = tdta, color = "Non-Default"), linewidth = 1) +
  geom_density(data = filter(y_X_train, yd == 1), aes(x = tdta, color = "Default"), linewidth = 1) +
  stat_function(fun = dnorm, args = list(mean = mean(X_train_safe$tdta, na.rm = TRUE),
                                         sd = sd(X_train_safe$tdta, na.rm = TRUE)),
                color = "darkgreen", linetype = "dashed") +
  stat_function(fun = dnorm, args = list(mean = mean(X_train_default$tdta, na.rm = TRUE),
                                         sd = sd(X_train_default$tdta, na.rm = TRUE)),
                color = "darkred", linetype = "dashed") +
  scale_fill_manual(values = c("Non-Default" = "green", "Default" = "red")) +
  scale_color_manual(values = c("Non-Default" = "green", "Default" = "red")) +
  labs(title = "Distribution of tdta for Default and Non-Default Groups", x = "tdta", y = "Density") +
  theme_minimal() +
  theme(legend.position = "top")
print(p)

# Jarque-Bera test for normality
numeric_cols <- names(y_X_train)[sapply(y_X_train, is.numeric) & names(y_X_train) != "yd"]
stats_list <- list()

for (yd in unique(y_X_train$yd)) {
  group_data <- filter(y_X_train, yd == !!yd)
  group_label <- ifelse(yd == 0, "NonDefault", "Default")
  for (col in numeric_cols) {
    data <- na.omit(group_data[[col]])
    n <- length(data)
    if (n > 0) {
      jb_test <- jarque.test(data)
      stats_list[[length(stats_list) + 1]] <- data.frame(
        Group = group_label,
        Variable = col,
        Obs = n,
        Skewness = round(skewness(data), 3),
        `Kurtosis-3` = round(kurtosis(data) - 3, 3),
        `JB Stat.` = round(jb_test$statistic, 3),
        `P-value` = ifelse(jb_test$p.value < 0.05, paste0(round(jb_test$p.value, 3), "*"), round(jb_test$p.value, 3))
      )
    }
  }
}

stats_df <- bind_rows(stats_list)
cat("Jarque-Bera X2 statistic H0: Skewness=0 and Kurtosis-3=0\n")
print(stats_df, row.names = FALSE)

# Levene's test for equal variances
results <- list()
for (var in names(X_train)) {
  group0 <- na.omit(X_train[y_X_train$yd == 0, var])
  group1 <- na.omit(X_train[y_X_train$yd == 1, var])
  n0 <- length(group0)
  n1 <- length(group1)
  if (n0 >= 2 & n1 >= 2) {
    levene_test <- leveneTest(y_X_train[[var]] ~ factor(y_X_train$yd))
    sd0 <- sd(group0, na.rm = TRUE)
    sd1 <- sd(group1, na.rm = TRUE)
    stat <- levene_test$`F value`[1]
    p <- levene_test$`Pr(>F)`[1]
    p_display <- ifelse(p < 0.05, paste0(round(p, 3), "*"), round(p, 3))
  } else {
    sd0 <- NA
    sd1 <- NA
    stat <- NA
    p <- NA
    p_display <- NA
  }
  results[[length(results) + 1]] <- data.frame(
    Variable = var,
    n0 = n0,
    sd0 = round(sd0, 3),
    n1 = n1,
    sd1 = round(sd1, 3),
    statistic = round(stat, 3),
    p_value = p_display
  )
}

df_results <- bind_rows(results)
print(df_results, row.names = FALSE)

# T-test for equality of means
results <- list()
for (var in names(X_train)) {
  group0 <- na.omit(X_train[y_X_train$yd == 0, var])
  group1 <- na.omit(X_train[y_X_train$yd == 1, var])
  n0 <- length(group0)
  n1 <- length(group1)
  if (n0 >= 2 & n1 >= 2) {
    mean0 <- mean(group0)
    mean1 <- mean(group1)
    mean_diff <- mean1 - mean0
    t_equal <- t.test(group0, group1, var.equal = TRUE)
    t_unequal <- t.test(group0, group1, var.equal = FALSE)
    p_equal_display <- ifelse(t_equal$p.value < 0.05, paste0(round(t_equal$p.value, 3), "*"), round(t_equal$p.value, 3))
    p_unequal_display <- ifelse(t_unequal$p.value < 0.05, paste0(round(t_unequal$p.value, 3), "*"), round(t_unequal$p.value, 3))
  } else {
    mean0 <- NA
    mean_diff <- NA
    t_stat_equal <- NA
    p_equal <- NA
    t_stat_unequal <- NA
    p_unequal <- NA
    p_equal_display <- NA
    p_unequal_display <- NA
  }
  results[[length(results) + 1]] <- data.frame(
    Variable = var,
    n0 = n0,
    m0 = round(mean0, 3),
    n1 = n1,
    `m1-m0` = round(mean_diff, 3),
    t_stat = round(t_equal$statistic, 3),
    p_value = p_equal_display,
    t_stat_dif = round(t_unequal$statistic, 3),
    p_value_dif = p_unequal_display
  )
}

df_results <- bind_rows(results)
cat("T-test for equality of means (yd=0 vs yd=1)\n")
print(df_results, row.names = FALSE)

# Linear Probability Model
y <- y_X_train$yd
X <- y_X_train %>% select(tdta) %>% add_column(const = 1, .before = 1)
lpm_model <- lm(yd ~ tdta, data = y_X_train)
summary(lpm_model)

# Scatter plot with regression line
p <- ggplot(y_X_train, aes(x = tdta, y = yd, color = factor(yd))) +
  geom_point(aes(shape = factor(yd)), alpha = 0.7) +
  geom_point(aes(y = predict(lpm_model), shape = factor(yd)), color = "black", shape = 4) +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  scale_color_manual(values = c("green", "red"), labels = c("Actual yd=0", "Actual yd=1")) +
  scale_shape_manual(values = c(16, 16), labels = c("Actual yd=0", "Actual yd=1")) +
  geom_hline(yintercept = c(0, 1), linetype = "dashed", color = "gray", alpha = 0.5) +
  labs(title = "Linear Probability Model using the train sample", x = "tdta", y = "yd (Binary)") +
  theme_minimal() +
  theme(legend.position = "top")
print(p)

# Comparison of LPM, Logit, and Probit models (single variable: tdta)
n0 <- sum(y == 0)
n1 <- sum(y == 1)

lpm_model <- lm(yd ~ tdta, data = y_X_train)
lpm_pred <- predict(lpm_model)
lpm_auc <- roc(y_X_train$yd, lpm_pred)$auc

logit_model <- glm(yd ~ tdta, family = binomial(link = "logit"), data = y_X_train)
logit_pred <- predict(logit_model, type = "response")
logit_auc <- roc(y_X_train$yd, logit_pred)$auc

probit_model <- glm(yd ~ tdta, family = binomial(link = "probit"), data = y_X_train)
probit_pred <- predict(probit_model, type = "response")
probit_auc <- roc(y_X_train$yd, probit_pred)$auc

# Create comparison table
results_table <- data.frame(
  `Linear Probability` = c(
    sprintf("%.3f (%.2f)", coef(lpm_model)[1], summary(lpm_model)$coefficients[1, 3]),
    sprintf("%.3f (%.2f)", coef(lpm_model)[2], summary(lpm_model)$coefficients[2, 3]),
    sprintf("%.3f", lpm_auc),
    sprintf("%.3f", summary(lpm_model)$r.squared),
    n0, n1
  ),
  Probit = c(
    sprintf("%.3f (%.2f)", coef(probit_model)[1], summary(probit_model)$coefficients[1, 3]),
    sprintf("%.3f (%.2f)", coef(probit_model)[2], summary(probit_model)$coefficients[2, 3]),
    sprintf("%.3f", probit_auc),
    sprintf("%.3f", 1 - probit_model$deviance/probit_model$null.deviance),
    n0, n1
  ),
  Logit = c(
    sprintf("%.3f (%.2f)", coef(logit_model)[1], summary(logit_model)$coefficients[1, 3]),
    sprintf("%.3f (%.2f)", coef(logit_model)[2], summary(logit_model)$coefficients[2, 3]),
    sprintf("%.3f", logit_auc),
    sprintf("%.3f", 1 - logit_model$deviance/logit_model$null.deviance),
    n0, n1
  ),
  row.names = c("Intercept", "tdta", "AUC", "R²/Pseudo-R²", "n₀", "n₁")
)
cat("Comparison of Linear Probability, Logit, and Probit Models\n")
cat("Parameter estimates with t-statistics in parentheses\n")
print(results_table)

# ROC curves for multiple variables
X_train_features <- X_train %>% select(tdta, gempl, opita, invsls, lsls)
X_test_features <- X_test %>% select(tdta, gempl, opita, invsls, lsls)

lpm_model <- lm(y_train ~ ., data = X_train_features)
lpm_pred <- predict(lpm_model, newdata = X_test_features)
lpm_roc <- roc(y_test, lpm_pred)

probit_model <- glm(y_train ~ ., family = binomial(link = "probit"), data = X_train_features)
probit_pred <- predict(probit_model, newdata = X_test_features, type = "response")
probit_roc <- roc(y_test, probit_pred)

logit_model <- glm(y_train ~ ., family = binomial(link = "logit"), data = X_train_features)
logit_pred <- predict(logit_model, newdata = X_test_features, type = "response")
logit_roc <- roc(y_test, logit_pred)

# Plot ROC curves
plot(lpm_roc, col = "#1f77b4", main = "ROC Curves Comparison")
plot(probit_roc, col = "#ff7f0e", add = TRUE)
plot(logit_roc, col = "#2ca02c", add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "black")
legend("bottomright", legend = c(
  sprintf("LPM (AUC = %.2f)", lpm_roc$auc),
  sprintf("Probit (AUC = %.2f)", probit_roc$auc),
  sprintf("Logit (AUC = %.2f)", logit_roc$auc)
), col = c("#1f77b4", "#ff7f0e", "#2ca02c"), lty = 1)

# Correlation matrix with t-statistics
corr_matrix <- cor(y_X_train, use = "pairwise.complete.obs")
n_samples <- nrow(y_X_train)
t_stats <- corr_matrix * sqrt(n_samples - 2) / sqrt(1 - corr_matrix^2)
annot <- matrix(sprintf("%.2f\n(%.2f)", corr_matrix, t_stats), nrow = nrow(corr_matrix))
mask <- upper.tri(corr_matrix, diag = TRUE)
corrplot(corr_matrix, method = "color", tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.7, number.digits = 2,
         diag = FALSE, type = "lower", col = colorRampPalette(c("red", "white", "red"))(200))

# Pair plots for first 7 variables
vars <- c("yd", "tdta", "reta", "opita", "ebita", "lsls", "lta", "gempl")
new_df <- y_X_train %>% mutate(target = factor(yd, levels = c(0, 1), labels = c("Non-Default", "Default")))
p <- ggpairs(new_df,
             columns = vars,
             aes(colour = target, fill = target),
             diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
             lower = list(continuous = wrap("points", alpha = 0.7) + 
                            geom_smooth(method = "lm", color = "blue", se = FALSE)),
             upper = list(continuous = "blank")) +
  scale_color_manual(values = c("green", "red")) +
  scale_fill_manual(values = c("green", "red")) +
  theme_minimal()
print(p)


