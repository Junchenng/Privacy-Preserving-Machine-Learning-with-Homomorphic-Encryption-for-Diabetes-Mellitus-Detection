# 2_model_training.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import loguniform, randint

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_configs():
    """Defines models and hyperparameter spaces."""
    return {
        'Logistic Regression': (LogisticRegression(), {'C': loguniform(1e-5, 100), 'solver': ['liblinear', 'lbfgs']}),
        'kNN': (KNeighborsClassifier(), {'n_neighbors': randint(1, 20)}),
        'SVM': (SVC(), {'C': loguniform(1e-5, 100), 'gamma': ['scale']}),
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': randint(1, 30)}),
        'Random Forest': (RandomForestClassifier(), {'n_estimators': randint(50, 500)}),
        'Ada Boost': (AdaBoostClassifier(), {'learning_rate': loguniform(0.001, 1)}),
        'Gradient Boosting': (GradientBoostingClassifier(), {'learning_rate': loguniform(0.001, 1), 'n_estimators': randint(50, 500)}),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'learning_rate': loguniform(0.01, 1)}),
        'Cat Boost': (CatBoostClassifier(verbose=0), {'learning_rate': loguniform(0.01, 0.3)})
    }

def train_models():
    # Load Data
    try:
        with open('processed_datasets.pkl', 'rb') as f:
            datasets = pickle.load(f)
    except FileNotFoundError:
        print("Run preprocessing first.")
        return

    # Use 'RUS' (Random Under Sampling) as it was preferred in your analysis
    # or loop through all. For brevity, we stick to RUS for the final model search.
    X, y = datasets.get('RUS', datasets['Original'])
    print(f"Training on RUS dataset (Shape: {X.shape})...")

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    configs = get_configs()

    for name, (model, params) in configs.items():
        print(f"Training {name}...")
        search = RandomizedSearchCV(model, params, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        search.fit(X_train_s, y_train)
        
        best = search.best_estimator_
        y_pred = best.predict(X_test_s)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Best_Estimator': best
        })

    # Save Results & Best Model
    df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    df_results.to_csv('model_results.csv', index=False)
    
    best_model = df_results.iloc[0]['Best_Estimator']
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    # Also save the specific RUS split for the Encryption script to verify
    with open('rus_split.pkl', 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)

    print(f"\nTraining Complete. Best Model: {df_results.iloc[0]['Model']}")
    return df_results

def plot_metrics(df):
    """Generates the 4 comparison bar charts from the notebook."""
    metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
    plt.figure(figsize=(20, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        ax = sns.barplot(x='Model', y=metric, data=df, palette='Set3')
        plt.title(metric)
        plt.xticks(rotation=45)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    print("Performance charts saved to 'model_performance_comparison.png'")

if __name__ == "__main__":
    results_df = train_models()
    plot_metrics(results_df)
