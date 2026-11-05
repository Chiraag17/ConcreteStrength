# Implementation Plan — Update App to Match New `Concrete.ipynb` Notebook

## 1. Overview
We will update the **Concrete Strength AI Lab** to include every table, mathematical equation, and visualization from the newly uploaded `Concrete.ipynb` notebook while preserving the clean **4-tab left sidebar** layout (`1. Dataset & EDA`, `2. PCA & Equations`, `3. Regression Comparison`, `4. Strength Predictor`).

---

## 2. Planned Changes by Tab

### Tab 1: Dataset & Exploratory Analysis (`DatasetExplorerSection.tsx`)
Matches **Cells 2–7** of `Concrete.ipynb`:
- **Dataset Shape & Missing Values Summary**: `(1030, 9)` dataset shape, `0` missing values across all columns, and train/test split shapes (`Training Data: (824, 8)`, `Testing Data: (206, 8)`).
- **First 5 Rows Preview (`df.head()`) & Statistical Summary (`df.describe()`)**: Clean toggle between the first 5 concrete samples and the full 8-row statistical summary (`count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`).
- **9×9 Correlation Heatmap (`sns.heatmap(df.corr(), annot=True, cmap="coolwarm")`)**: Annotated 9×9 correlation matrix showing exact correlation values across all 8 input ingredients plus `Concrete compressive strength`.
- **Concrete Strength Distribution (`sns.histplot(..., kde=True)`)**: Histogram with overlaid smooth KDE line showing `Strength (MPa)` vs `Frequency`.

### Tab 2: PCA Simulator, Coefficients & Equations (`PCASimulatorSection.tsx`)
Matches **Cells 8–15** of `Concrete.ipynb`:
- **PCA Summary Metrics**: `Original Features: 8`, `Selected Components: 6`, `Variance Retained: 97.31%`, `Dimensions Reduced: 2`, `Before PCA: (824, 8)`, `After PCA: (824, 6)`.
- **Cumulative Explained Variance Chart & Table (`variance_table`)**:
  - Interactive `k` selector (`k = 1..8`, defaulting to `k = 6` at the 95% threshold).
  - Line chart of `Cumulative Explained Variance (%)` with the red dashed `95% Variance` threshold line.
  - Full `variance_table` showing `Principal Component` (`PC1`–`PC8`), `Explained Variance (%)`, and `Cumulative Variance (%)`.
- **NEW — PCA Component Coefficients Table (`pca_coefficients`)**:
  - Displays the exact `PC1`–`PC6` (or up to selected `k`) loading coefficients across all 8 standardized features (`Cement`, `Blast Furnace Slag`, `Fly Ash`, `Water`, `Superplasticizer`, `Coarse Aggregate`, `Fine Aggregate`, `Age`), rounded to 3 decimal places with color-coded positive/negative weights.
- **NEW — PCA Mathematical Equations**:
  - Displays the explicit linear combination equations for each principal component (`PC1` through `PC6`) exactly as printed in Cell 14 of the notebook:
    $$\text{PC}_i = \sum_{j=1}^{8} w_{ij} \times \text{Feature}_j$$
- **NEW — PCA Transformed Dataset Preview (`pca_df.head()`)**:
  - Displays the first 5 transformed training rows across `PC1`–`PC6` (`[-1.175582, 1.221243, ...]`, etc.) from Cell 15.

### Tab 3: Regression Comparison (`RegressionComparisonSection.tsx`)
Matches **Cells 16–23** of `Concrete.ipynb`:
- **Model Performance Matrix Table**:
  - Side-by-side comparison of `Without PCA` vs `With PCA`:
    - `Input features`: `8` vs `6`
    - `Variance retained (%)`: `100.00%` vs `97.31%`
    - `MAE`: `7.7454` vs `8.4876`
    - `MSE`: `95.9755` vs `111.3343`
    - `RMSE`: `9.7967` vs `10.5515`
    - `R2 Score`: `0.6275` vs `0.5679`
- **NEW — 2×2 Regression Performance Comparison Bar Charts**:
  - Four dedicated subplot cards (`MAE`, `MSE`, `RMSE`, and `R2 Score`) comparing `Without PCA` vs `With PCA` just like Cell 19's `2×2` matplotlib figure.
- **Actual vs. Predicted Scatter Plot (`Regression Comparison`)**:
  - Scatter plot comparing `Without PCA` and `With PCA` test predictions against the diagonal `Perfect Prediction` line.
- **Final Results Summary Banner**:
  - Highlights the notebook's conclusion: **25.0% dimensionality reduction** (`8 → 6` features) while retaining **97.31% variance**, noting how PCA slightly reduces linear regression accuracy (`R² = 0.6275` vs `0.5679`).

### Tab 4: Strength Predictor (`PredictorAndLabSection.tsx`)
- Preserves the interactive **Concrete Strength Predictor** where users can adjust the 8 mix sliders or load sample mixes to compare real-time `Without PCA` vs `With PCA` MPa predictions and inspect the live `PC1`–`PC6` transformed values for that custom mix.
