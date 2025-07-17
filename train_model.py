import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Load Data ====
file_path = 'international_matches.csv'
df = pd.read_csv(file_path)

# ==== Data Preprocessing ====
# Convert date to datetime and extract year
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Create target variable (home team result)
df['home_team_result'] = df['home_team_result'].map({'Win': 2, 'Draw': 1, 'Lose': 0})

# Select relevant features
features = [
    'home_team_fifa_rank', 'away_team_fifa_rank',
    'home_team_total_fifa_points', 'away_team_total_fifa_points',
    'home_team_score', 'away_team_score',
    'home_team_goalkeeper_score', 'away_team_goalkeeper_score',
    'home_team_mean_defense_score', 'away_team_mean_defense_score',
    'home_team_mean_offense_score', 'away_team_mean_offense_score',
    'home_team_mean_midfield_score', 'away_team_mean_midfield_score',
    'year'
]

# Handle missing values (simple imputation)
df[features] = df[features].fillna(df[features].mean())

X = df[features]
y = df['home_team_result']

# ==== Train-Test Split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ==== Feature Scaling ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==== Handle Class Imbalance ====
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# ==== Models ====
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

params = {
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    "XGBoost": {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\n🚀 Training and evaluating: {name}")

    grid = GridSearchCV(model, params[name], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)

    best_estimator = grid.best_estimator_
    y_pred = best_estimator.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc*100:.2f}% with best params: {grid.best_params_}")
    print(classification_report(y_test, y_pred, target_names=['Lose', 'Draw', 'Win']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Lose', 'Draw', 'Win'], 
                yticklabels=['Lose', 'Draw', 'Win'])
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Feature Importance
    if hasattr(best_estimator, 'feature_importances_'):
        feat_importance = pd.Series(best_estimator.feature_importances_, 
        index=features).sort_values(ascending=False)
        print(f"Feature Importance for {name}:\n{feat_importance}\n")
        plt.figure(figsize=(10, 6))
        feat_importance.plot(kind='bar', title=f'Feature Importance: {name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = best_estimator
        best_model_name = name

# ==== Save Best Model & Scaler ====
joblib.dump(best_model, 'football_match_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\n✅✅ Best model: {best_model_name} with accuracy {best_accuracy*100:.2f}%")
print("✅ Model and Scaler saved successfully!")