# 🏡 Ames Housing Price Prediction

**An End-to-End Machine Learning Regression Project**

A machine learning project focused on predicting residential house sale prices using the Ames Housing dataset. This project explores data characteristics, investigates relationships between features and sale prices, and compares regression models with and without Principal Component Analysis (PCA).

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Modeling Approaches](#-modeling-approaches)
- [Model Comparison](#-model-comparison)
- [Evaluation Metrics](#-evaluation-metrics)
- [Model Selection](#-model-selection)
- [Model Persistence](#-model-persistence)
- [Repository Structure](#-repository-structure)
- [Installation and Usage](#-installation-and-usage)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

## 🎯 Project Overview

The goal of this project is to develop regression models that estimate the sale price of residential properties from their characteristics.

The project focuses on:

- Exploring the dataset and understanding its features.
- Examining the target variable and relationships between features.
- Preparing the data for machine learning.
- Training and evaluating multiple regression algorithms.
- Tuning model hyperparameters with `GridSearchCV`.
- Comparing models trained with and without PCA.
- Selecting representative models based on test-set performance.
- Saving fitted models for future use with Joblib.

**Problem type:** Supervised regression

**Target variable:** `Sale_Price`

**Primary evaluation criteria:** Test RMSE and Test R²

## 📊 Dataset

The project retrieves the Ames Housing dataset from OpenML using `fetch_openml`.

- **Source:** [OpenML — Ames Housing](https://www.openml.org/d/41211)
- **Dataset ID:** `41211`
- **Observations:** 2,930 rows
- **Columns:** 81
- **Target:** `Sale_Price`

The dataset contains numerical and categorical information describing residential properties in Ames, Iowa.

The target variable represents the sale price of each property. The project uses these historical observations to learn patterns that can help estimate prices for unseen properties.

## 🔄 Project Workflow

### 1. Data Acquisition and Inspection

The dataset is loaded from OpenML into a Pandas DataFrame. Initial inspection examines the dataset's dimensions, column names, data types, and general structure.

### 2. Exploratory Data Analysis (EDA)

Exploratory analysis investigates the distribution of `Sale_Price` and relationships between property characteristics and the target.

Visualizations and statistical summaries help identify patterns, unusual observations, and potentially useful features.

### 3. Feature Analysis

Feature relationships with the target are investigated using correlation analysis where applicable. This provides an initial view of numerical relationships but does not establish causation.

### 4. Data Preparation

The features and target are separated:

- `X`: property features
- `y`: `Sale_Price`

The data is divided into training and test sets using an 80/20 split with `random_state=42`.

For the non-PCA modeling approach, the training features are scaled using `StandardScaler`, and the fitted scaler is used to transform the test features.

### 5. Model Training and Hyperparameter Tuning

Multiple regression algorithms are evaluated. `GridSearchCV` with five-fold cross-validation and R² scoring is used in the modeling experiments to identify suitable hyperparameters.

### 6. Evaluation and Comparison

The models are compared using training and test metrics. Two approaches are investigated:

- Modeling without PCA.
- Modeling with PCA.

The detailed experiment implementations are documented in their respective Git branches.

### 7. Final Model Selection and Persistence

The final models are selected based on their test-set performance and saved using Joblib for future reuse.

## 🧠 Modeling Approaches

### Approach A — Without PCA

This approach retains the processed feature space without applying PCA.

The experiments include the following regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- HistGradientBoosting

The models are evaluated using a consistent train/test evaluation strategy within the experiment.

**Experiment branch:** [Modeling Without PCA](https://github.com/Mohamadzf23/ames-housing-price-prediction/tree/modeling-without-pca)

### Approach B — With PCA

This approach applies Principal Component Analysis after feature scaling to reduce dimensionality while retaining 95% of the variance for the ElasticNet pipeline.

The evaluated models include:

- ElasticNet
- Ridge Regression
- Lasso Regression
- Gradient Boosting
- HistGradientBoosting

The experiments investigate how dimensionality reduction affects predictive performance and the difference between training and test results.

**Experiment branch:** [Modeling With PCA](https://github.com/Mohamadzf23/ames-housing-price-prediction/tree/modeling-with-pca)

## 📈 Model Comparison

The following table summarizes the selected model from each approach using the recorded test-set results.

| Approach | Selected Model | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|
| Without PCA | Gradient Boosting | 11,302.61 | 15,652.86 | 0.9212 |
| With PCA | ElasticNet | 13,680.91 | 18,977.92 | 0.8842 |

*These values represent the recorded results for the current train/test split. They are not a guarantee of performance on future or external data.*

### Results Interpretation

- **Gradient Boosting without PCA** achieved the lowest Test RMSE and highest Test R² among the evaluated models overall.
- **ElasticNet with PCA** was selected as the strongest model within the evaluated PCA-based group.
- The non-PCA Gradient Boosting model performed better on the current test split according to both RMSE and R².
- PCA may simplify the feature space, but dimensionality reduction does not necessarily improve predictive accuracy.

The full experimental details, preprocessing choices, and hyperparameter tuning are available in the two modeling branches linked above.

## 📐 Evaluation Metrics

Three regression metrics are used to assess predictive performance.

| Metric | Meaning | Preferred Direction |
|---|---|---|
| MAE | Mean Absolute Error; the average absolute prediction error | Lower |
| RMSE | Root Mean Squared Error; penalizes larger errors more strongly than MAE | Lower |
| R² | Coefficient of Determination; measures how much target variation is explained by the model | Higher |

RMSE and MAE are expressed in the same price units as `Sale_Price`. R² is unitless.

Training metrics help assess how well a model fits the training data, while test metrics provide an estimate of performance on held-out observations.

## 💾 Model Persistence

The selected models are saved with Joblib:

| Model | Saved File | Preprocessing |
|---|---|---|
| Gradient Boosting — Without PCA | `gradient_boosting_without_pca.joblib` | Requires the same fitted scaling transformation used during training |
| ElasticNet — With PCA | `elasticnet_with_pca.joblib` | Includes `StandardScaler`, `PCA`, and `ElasticNet` in a fitted pipeline |

The complete ElasticNet pipeline preserves its fitted preprocessing steps. The Gradient Boosting estimator was trained on scaled features, so its fitted scaler must also be preserved and reused when predicting on new observations.

For reproducible predictions, new data must have compatible feature columns and ordering, and must undergo the same preprocessing used during training.

## 📁 Repository Structure

The currently visible repository structure includes the `src` directory, `.gitignore`, `LICENSE`, and `README.md`.

The intended organization for the complete project is:

```text
ames-housing-price-prediction/
├── src/
│   └── ames_housing_price_prediction.ipynb
├── models/
│   ├── gradient_boosting_without_pca.joblib
│   └── elasticnet_with_pca.joblib
├── model_comparison.csv
├── results_without_pca.csv
├── results_with_pca.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The `models/` directory and comparison CSV files should be added to the repository if you want them to be available to other users. The structure above describes the intended organization; not every listed file was present in the inspected root directory.

## 🚀 Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Mohamadzf23/ames-housing-price-prediction.git
cd ames-housing-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the main notebook:

`src/ames_housing_price_prediction.ipynb`

Run its cells in order to inspect the dataset, explore the analysis, and review the model comparison.

The notebook retrieves the dataset from OpenML, so access to the dataset source is required when running the data-loading step.

## ⚠️ Limitations

- **Geographic scope:** The data describes properties in Ames, Iowa. The models should not be assumed to generalize directly to other real-estate markets.
- **Historical data:** Predictions may not reflect changes in market conditions, inflation, or buyer preferences.
- **Test-set dependence:** Reported metrics depend on the selected train/test split and may vary with a different split.
- **Preprocessing consistency:** The saved Gradient Boosting estimator requires the same fitted scaler and feature representation used during training.
- **Model artifacts:** Reproducing predictions requires the saved model files, compatible dependencies, and the appropriate preprocessing objects.

## 🔮 Future Improvements

- Preserve and version all fitted preprocessing objects alongside the models.
- Add a reproducible prediction script for new property records.
- Analyze residuals and prediction errors across different price ranges.
- Investigate feature importance and model interpretability.
- Add automated checks for input feature names, ordering, and missing values.
- Record exact dependency versions to improve reproducibility.

## 👤 Author

**Mohamadreza Zafari**

- [GitHub](https://github.com/Mohamadzf23)
- [LinkedIn](https://www.linkedin.com/in/mohamadzf23/)

---

*This project was developed for learning, experimentation, and portfolio purposes. Model predictions are estimates based on historical data and should not be treated as guaranteed property valuations.*