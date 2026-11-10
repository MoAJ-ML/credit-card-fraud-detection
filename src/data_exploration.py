import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_and_explore_data(data_path='data/creditcard.csv', viz_dir='visualizations', summary_file='summary.txt'):
    """
    Loads the dataset, prints basic info, saves all visualizations, and writes a summary of findings.
    """
    if not os.path.exists(data_path) and not os.path.exists('archive (2).zip'):
        raise FileNotFoundError(f"Dataset not found at {data_path} or in archive (2).zip")
    os.makedirs(viz_dir, exist_ok=True)
    summary_lines = []
    df = pd.read_csv(data_path if os.path.exists(data_path) else 'archive (2).zip', compression='infer')
    # Format 'Time' as hours since first transaction
    df['Hour'] = ((df['Time'] // 3600) % 24).astype(int)  # Hour of day (0-23)
    # Dataset shape
    summary_lines.append(f"Dataset shape: {df.shape}\n")
    # Columns and data types
    summary_lines.append("Columns and data types:\n" + str(df.dtypes) + "\n")
    # Summary statistics
    summary_lines.append("Summary statistics:\n" + str(df.describe().T) + "\n")
    # Class distribution
    class_counts = df['Class'].value_counts()
    summary_lines.append(f"Class distribution:\n{class_counts}\n")
    plt.figure(figsize=(8,6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title('Class Distribution (0 = Non-Fraud, 1 = Fraud)', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    # Missing/duplicated
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    summary_lines.append(f"Total missing values: {missing}\n")
    summary_lines.append(f"Duplicated rows: {duplicates}\n")
    # Amount distribution
    plt.figure(figsize=(10,6))
    sns.histplot(df['Amount'], bins=50, kde=True, color='steelblue')
    plt.title('Transaction Amount Distribution', fontsize=16)
    plt.xlabel('Amount', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'amount_distribution.png'), dpi=150)
    plt.close()
    # Time (Hour) distribution
    plt.figure(figsize=(10,6))
    # Ensure only 0-23 are shown, with one bin per hour
    sns.histplot(df['Hour'], bins=np.arange(25)-0.5, kde=False, color='orange', edgecolor='black')
    plt.title('Transaction Hour of Day Distribution', fontsize=16)
    plt.xlabel('Hour of Day (0–23)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(np.arange(0, 24, 1), fontsize=12)
    plt.xlim(-0.5, 23.5)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'hour_distribution.png'), dpi=150)
    plt.close()
    # Correlation heatmap (top features)
    corr = df.corr()
    top_corr_features = corr['Class'].abs().sort_values(ascending=False).head(6).index
    plt.figure(figsize=(10,8))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size":12})
    plt.title('Top Correlated Features with Class', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top_correlated_features.png'), dpi=150)
    plt.close()
    # Add findings to summary
    summary_lines.append("\nKey Findings from Data Exploration:\n")
    summary_lines.append(f"- The dataset contains {df.shape[0]} transactions, with {class_counts[1]} ({class_counts[1]/df.shape[0]*100:.4f}%) labeled as fraud.\n")
    summary_lines.append(f"- No missing values detected. {duplicates} duplicated rows found.\n")
    summary_lines.append("- Transaction amounts are highly skewed, with most transactions being low value.\n")
    summary_lines.append("- Fraudulent transactions are extremely rare, highlighting severe class imbalance.\n")
    summary_lines.append("- The 'Hour' variable (derived from 'Time') shows transaction activity by hour of the day (0–23).\n")
    summary_lines.append("- Top features most correlated with fraud: " + ', '.join(top_corr_features.drop('Class')) + "\n")
    # Save summary
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nData exploration complete. Visualizations saved to '{viz_dir}/', summary saved to '{summary_file}'.\n")
    return df
