"""
=============================================================================
Predictive Maintenance - Industrial Equipment Failure Forecasting
=============================================================================
Full ML pipeline: data generation -> cleaning -> EDA -> feature engineering ->
model training -> hyperparameter tuning -> SHAP explainability -> model export.

Author : Predictive Maintenance Project
Date   : 2024
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# Force UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Paths ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sensor_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================================================================
# STEP 1 -- SYNTHETIC DATA GENERATION
# =========================================================================
def generate_sensor_data(n_rows: int = 100_000, n_machines: int = 50) -> pd.DataFrame:
    """
    Generate realistic industrial sensor data with physics-informed failure
    logic.  A composite risk score drives the failure probability through a
    sigmoid function, producing a ~5-8 % failure rate.
    """
    print("=" * 70)
    print("  STEP 1 -- Generating Synthetic Sensor Data")
    print("=" * 70)

    rows_per_machine = n_rows // n_machines
    records = []

    for i in range(n_machines):
        machine_id = f"M{i + 1:03d}"
        # Each machine starts at a random point in time (past ~83 days)
        base_time = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=np.random.randint(0, 48))
        timestamps = pd.date_range(start=base_time, periods=rows_per_machine, freq="h")

        # Base operating hours -- varies per machine (older vs newer)
        base_hours = np.random.uniform(500, 15000)
        operating_hours = base_hours + np.arange(rows_per_machine)

        # --- Sensor readings with realistic drift & noise ---
        temperature = np.random.normal(75, 10, rows_per_machine)
        # Slight upward drift for older machines
        temperature += (operating_hours - base_hours) * 0.0005
        temperature = np.clip(temperature, 60, 110)

        vibration = np.random.normal(2.5, 1.2, rows_per_machine)
        vibration = np.clip(vibration, 0.5, 8.0)

        pressure = np.random.normal(55, 8, rows_per_machine)
        # Pressure tends to drop as equipment ages
        pressure -= (operating_hours - base_hours) * 0.0003
        pressure = np.clip(pressure, 25, 75)

        humidity = np.random.normal(50, 10, rows_per_machine)
        humidity = np.clip(humidity, 30, 80)

        voltage = np.random.normal(230, 5, rows_per_machine)
        voltage = np.clip(voltage, 210, 250)

        current = np.random.normal(12, 3, rows_per_machine)
        current = np.clip(current, 5, 20)

        rotational_speed = np.random.normal(2500, 500, rows_per_machine)
        rotational_speed = np.clip(rotational_speed, 1000, 4000)

        # --- Composite risk score (physics-informed) ---
        risk = (
            0.30 * ((temperature - 60) / 50)          # Higher temp -> higher risk
            + 0.25 * ((vibration - 0.5) / 7.5)         # Higher vibration -> higher risk
            + 0.20 * ((75 - pressure) / 50)             # Lower pressure -> higher risk
            + 0.10 * (np.abs(voltage - 230) / 20)       # Voltage deviation -> higher risk
            + 0.15 * ((operating_hours - 500) / 19500)  # More hours -> higher risk
        )

        # Sigmoid with offset to target ~5-8 % failure rate
        failure_prob = 1 / (1 + np.exp(-12 * (risk - 0.55)))
        failure = (np.random.rand(rows_per_machine) < failure_prob).astype(int)

        machine_df = pd.DataFrame({
            "machine_id": machine_id,
            "timestamp": timestamps,
            "temperature": np.round(temperature, 2),
            "vibration": np.round(vibration, 3),
            "pressure": np.round(pressure, 2),
            "humidity": np.round(humidity, 2),
            "voltage": np.round(voltage, 2),
            "current": np.round(current, 2),
            "rotational_speed": np.round(rotational_speed, 1),
            "operating_hours": np.round(operating_hours, 1),
            "failure": failure,
        })
        records.append(machine_df)

    df = pd.concat(records, ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    failure_rate = df["failure"].mean() * 100
    print(f"  [OK] Generated {len(df):,} rows for {n_machines} machines")
    print(f"  [OK] Failure rate: {failure_rate:.2f}%")
    print(f"  [OK] Saved to: {DATA_PATH}\n")
    return df


# =========================================================================
# STEP 2 -- DATA LOADING
# =========================================================================
def load_data() -> pd.DataFrame:
    """Load sensor data from CSV with proper dtypes."""
    print("=" * 70)
    print("  STEP 2 -- Loading Data")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  [OK] Loaded {len(df):,} rows x {len(df.columns)} columns")
    print(f"  [OK] Columns: {list(df.columns)}")
    print(f"  [OK] Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}\n")
    return df


# =========================================================================
# STEP 3 -- DATA CLEANING
# =========================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and outliers."""
    print("=" * 70)
    print("  STEP 3 -- Data Cleaning")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "failure"]

    # --- Inject ~1 % missing values for realism, then impute ---
    n_missing = int(0.01 * df.shape[0] * len(numeric_cols))
    rows_idx = np.random.choice(df.index, n_missing, replace=True)
    cols_idx = np.random.choice(numeric_cols, n_missing, replace=True)
    for r, c in zip(rows_idx, cols_idx):
        df.at[r, c] = np.nan

    missing_before = df.isnull().sum().sum()
    print(f"  * Missing values injected: {missing_before}")

    # Impute with median (assignment required for pandas >= 3.0)
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print(f"  [OK] Missing values after imputation: {df.isnull().sum().sum()}")

    # --- Remove duplicates ---
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"  [OK] Duplicates removed: {dups}")

    # --- Outlier treatment (IQR clipping) ---
    outliers_clipped = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outliers_clipped += mask.sum()
        df[col] = df[col].clip(lower, upper)

    print(f"  [OK] Outlier values clipped: {outliers_clipped}")
    print(f"  [OK] Final shape: {df.shape}\n")
    return df.reset_index(drop=True)


# =========================================================================
# STEP 4 -- EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================================
def perform_eda(df: pd.DataFrame) -> None:
    """Generate and save EDA plots."""
    print("=" * 70)
    print("  STEP 4 -- Exploratory Data Analysis")
    print("=" * 70)

    # 1. Failure distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["failure"].value_counts().sort_index()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(["Normal (0)", "Failure (1)"], counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title("Failure Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "failure_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: plots/failure_distribution.png")

    # 2. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: plots/correlation_heatmap.png")

    # 3. Preliminary feature importance (Random Forest)
    feature_cols = [c for c in numeric_df.columns if c != "failure"]
    X_temp = df[feature_cols].copy()
    y_temp = df["failure"].copy()
    rf_temp = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    rf_temp.fit(X_temp, y_temp)

    importances = pd.Series(rf_temp.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind="barh", color="#3498db", edgecolor="white", ax=ax)
    ax.set_title("Feature Importance (Random Forest)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Importance", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: plots/feature_importance.png\n")


# =========================================================================
# STEP 5 -- FEATURE ENGINEERING
# =========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling, lag, and rate-of-change features."""
    print("=" * 70)
    print("  STEP 5 -- Feature Engineering")
    print("=" * 70)

    # Sort by machine and time for correct windowing
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    sensor_cols = ["temperature", "vibration", "pressure"]

    new_features = []

    for col in sensor_cols:
        # Rolling mean & std (window = 5)
        df[f"{col}_rolling_mean_5"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
        df[f"{col}_rolling_std_5"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(window=5, min_periods=1).std())
        )
        new_features.extend([f"{col}_rolling_mean_5", f"{col}_rolling_std_5"])

    # Lag features
    for col in ["temperature", "vibration"]:
        for lag in [1, 3]:
            df[f"{col}_lag_{lag}"] = df.groupby("machine_id")[col].shift(lag)
            new_features.append(f"{col}_lag_{lag}")

    # Rate of change (diff)
    for col in sensor_cols:
        df[f"{col}_rate_of_change"] = df.groupby("machine_id")[col].diff()
        new_features.append(f"{col}_rate_of_change")

    # Fill NaNs from shift / diff / rolling with 0
    df[new_features] = df[new_features].fillna(0)

    print(f"  [OK] Created {len(new_features)} new features:")
    for feat in new_features:
        print(f"      - {feat}")
    print(f"  [OK] Final shape: {df.shape}\n")
    return df


# =========================================================================
# STEP 6 -- CLASS IMBALANCE HANDLING (SMOTE)
# =========================================================================
def handle_imbalance(X_train, y_train):
    """Apply SMOTE to balance the training set."""
    print("=" * 70)
    print("  STEP 6 -- Class Imbalance Handling (SMOTE)")
    print("=" * 70)

    print(f"  Before SMOTE:")
    print(f"      Class 0 (Normal):  {(y_train == 0).sum():,}")
    print(f"      Class 1 (Failure): {(y_train == 1).sum():,}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n  After SMOTE:")
    print(f"      Class 0 (Normal):  {(y_resampled == 0).sum():,}")
    print(f"      Class 1 (Failure): {(y_resampled == 1).sum():,}")
    print()
    return X_resampled, y_resampled


# =========================================================================
# STEP 7 -- MODEL TRAINING & COMPARISON
# =========================================================================
def train_and_compare(X_train, y_train, X_test, y_test):
    """Train Logistic Regression, Random Forest, and XGBoost; compare metrics."""
    print("=" * 70)
    print("  STEP 7 -- Model Training & Comparison")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=1),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=1,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...", end=" ")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        results[name] = metrics
        trained_models[name] = model
        print("Done [OK]")

    # Print comparison table
    print("\n  +-------------------------+----------+-----------+--------+----------+---------+")
    print("  | Model                   | Accuracy | Precision | Recall | F1 Score | ROC-AUC |")
    print("  +-------------------------+----------+-----------+--------+----------+---------+")
    for name, m in results.items():
        print(f"  | {name:<23} | {m['Accuracy']:.4f}   | {m['Precision']:.4f}    | {m['Recall']:.4f} | {m['F1 Score']:.4f}   | {m['ROC-AUC']:.4f}  |")
    print("  +-------------------------+----------+-----------+--------+----------+---------+")

    # Select best model by F1
    best_name = max(results, key=lambda k: results[k]["F1 Score"])
    print(f"\n  >> Best Model: {best_name} (F1 = {results[best_name]['F1 Score']:.4f})\n")

    return trained_models, results, best_name


# (Hyperparameter tuning removed for speed)


# =========================================================================
# STEP 9 -- SHAP EXPLAINABILITY
# =========================================================================
def generate_shap_plots(model, X_test, feature_names):
    """Generate SHAP summary and feature importance plots."""
    print("=" * 70)
    print("  STEP 9 -- SHAP Explainability")
    print("=" * 70)

    explainer = shap.TreeExplainer(model)
    # Use a subsample for speed if dataset is large
    sample_size = min(2000, len(X_test))
    X_sample = X_test[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.title("SHAP Summary Plot", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: plots/shap_summary.png")

    # SHAP Feature Importance (bar)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved: plots/shap_feature_importance.png\n")


# =========================================================================
# STEP 10 -- SAVE BEST MODEL & SCALER
# =========================================================================
def save_artifacts(model, scaler):
    """Save trained model and scaler to disk."""
    print("=" * 70)
    print("  STEP 10 -- Saving Artifacts")
    print("=" * 70)

    joblib.dump(model, MODEL_PATH)
    print(f"  [OK] Model saved: {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"  [OK] Scaler saved: {SCALER_PATH}\n")


# =========================================================================
# STEP 11 -- PRINT FINAL METRICS
# =========================================================================
def print_final_metrics(model_name, metrics):
    """Display the final model performance summary."""
    print("=" * 70)
    print("  STEP 11 -- FINAL RESULTS")
    print("=" * 70)
    print(f"\n  >> Best Model: {model_name}")
    print(f"  ------------------------------------")
    for metric, value in metrics.items():
        print(f"    {metric:<12}: {value:.4f}")
    print(f"  ------------------------------------")
    print(f"\n  [OK] All artifacts saved successfully!")
    print(f"  [OK] Run `streamlit run streamlit_app.py` to launch the dashboard.\n")


# =========================================================================
# MAIN PIPELINE
# =========================================================================
def main():
    print("\n" + "#" * 70)
    print("  PREDICTIVE MAINTENANCE -- ML PIPELINE")
    print("#" * 70 + "\n")

    # Step 1: Generate data if it doesn't exist
    if not os.path.exists(DATA_PATH):
        generate_sensor_data()

    # Step 2: Load data
    df = load_data()

    # Step 3: Clean data
    df = clean_data(df)

    # Step 4: EDA
    perform_eda(df)

    # Step 5: Feature engineering
    df = engineer_features(df)

    # -- Prepare features & target --
    drop_cols = ["machine_id", "timestamp", "failure"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df["failure"].values

    # Safety: fill any remaining NaN values with 0
    X = np.nan_to_num(X, nan=0.0)

    # Train-test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 6: SMOTE on training data
    X_train_resampled, y_train_resampled = handle_imbalance(X_train_scaled, y_train)

    # Step 7: Train and compare models
    trained_models, results, best_name = train_and_compare(
        X_train_resampled, y_train_resampled, X_test_scaled, y_test
    )

    # Final Model Selection
    final_model = trained_models[best_name]
    final_name = best_name
    final_metrics = results[best_name]

    # Step 8: SHAP
    # Always use the XGBoost model for SHAP plots
    shap_model = trained_models["XGBoost"]
    generate_shap_plots(shap_model, X_test_scaled, feature_cols)

    # Step 10: Save artifacts
    save_artifacts(final_model, scaler)

    # Save feature columns for the dashboard
    joblib.dump(feature_cols, os.path.join(BASE_DIR, "feature_cols.pkl"))

    # Step 11: Final summary
    print_final_metrics(final_name, final_metrics)


if __name__ == "__main__":
    main()
