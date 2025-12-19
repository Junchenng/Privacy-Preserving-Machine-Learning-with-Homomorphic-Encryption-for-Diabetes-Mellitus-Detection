import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE

def clean_data(filepath='diabetes.csv'):
    """Loads data, handles zeros, and removes outliers."""
    print(f"Loading {filepath}...")
    data = pd.read_csv(filepath)
    
    # Replace zeros with mean for specific columns
    cols_to_fix = ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']
    for col in cols_to_fix:
        data[col] = data[col].replace(0, data[col].mean())

    # Remove outliers based on quantiles from the notebook
    quantiles = {
        'Pregnancies': 0.98, 'SkinThickness': 0.99, 'Insulin': 0.95,
        'BMI': 0.99, 'DiabetesPedigreeFunction': 0.99, 'Age': 0.99
    }
    initial_len = len(data)
    for col, q in quantiles.items():
        outlier_val = data[col].quantile(q)
        data = data[data[col] < outlier_val]
    
    print(f"Cleaned data: {len(data)} rows (removed {initial_len - len(data)} outliers)")
    return data

def generate_resampled_datasets(data):
    """Generates a dictionary containing multiple versions of the dataset."""
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    datasets = {'Original': (X, y)}
    
    # Define all samplers used in the notebook
    samplers = {
        'RUS': RandomUnderSampler(random_state=42, replacement=True),
        'ROS': RandomOverSampler(random_state=42),
        'Tomek': TomekLinks(sampling_strategy='majority'),
        'SMOTE': SMOTE(random_state=42),
        'NearMiss': NearMiss()
    }

    print("Generating resampled datasets...")
    for name, sampler in samplers.items():
        X_res, y_res = sampler.fit_resample(X, y)
        datasets[name] = (X_res, y_res)
        print(f" - {name}: {Counter(y_res)}")
    
    return datasets

def visualize_data(datasets):
    """Generates PCA scatter plots and Bar charts for each resampling technique."""
    print("Generating resampling visualizations...")
    n = len(datasets)
    fig, axes = plt.subplots(n, 2, figsize=(15, 5 * n))
    plt.subplots_adjust(hspace=0.4)

    for i, (name, (X, y)) in enumerate(datasets.items()):
        # 1. Class Distribution Bar Chart
        count = Counter(y)
        sns.barplot(x=list(count.keys()), y=list(count.values()), ax=axes[i, 0], palette='viridis')
        axes[i, 0].set_title(f'{name}: Class Distribution')
        axes[i, 0].set_xlabel('Outcome')
        
        # 2. PCA 2D Projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter = axes[i, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, edgecolor='k')
        axes[i, 1].set_title(f'{name}: PCA Projection')
        axes[i, 1].legend(*scatter.legend_elements(), title="Outcome")

    plt.savefig('resampling_visualization.png')
    print("Visualization saved to 'resampling_visualization.png'")

if __name__ == "__main__":
    df = clean_data()
    datasets = generate_resampled_datasets(df)
    
    # Visualization
    visualize_data(datasets)
    
    # Save datasets for the next steps
    with open('processed_datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    print("Preprocessing complete. Data saved.")
