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
    # Holdout must be isolated before fitting any data-dependent transform.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    # Resample training rows only. Any future cross-validation must put this
    # transform and SMOTE inside each training fold.
    minority_count = int(y_train.value_counts().min())
    if minority_count < 2:
        raise ValueError('SMOTE needs at least two minority samples in the training set')
    smote = SMOTE(random_state=random_state, k_neighbors=min(5, minority_count - 1))
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"\nOriginal training set class distribution:\n{y_train.value_counts()}")
    print(f"\nBalanced training set class distribution (SMOTE):\n{y_train_bal.value_counts()}")
    return X_train, X_test, y_train, y_test, X_train_bal, y_train_bal
