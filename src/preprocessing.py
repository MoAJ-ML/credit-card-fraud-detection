import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Handles class imbalance, scaling, and train-test split.
    Returns: X_train, X_test, y_train, y_test, X_train_bal, y_train_bal
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    # Scaling
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state)
    # SMOTE for balancing
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"\nOriginal training set class distribution:\n{y_train.value_counts()}")
    print(f"\nBalanced training set class distribution (SMOTE):\n{y_train_bal.value_counts()}")
    return X_train, X_test, y_train, y_test, X_train_bal, y_train_bal
