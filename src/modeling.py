from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV

def train_and_compare_models(X_train, X_test, y_train, y_test, X_train_bal, y_train_bal):
    """
    Train and compare Logistic Regression, Random Forest, and XGBoost on both imbalanced and balanced data.
    Returns: models dict and results dict
    """
    models = {}
    results = {}
    # Logistic Regression
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_bal = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    lr_bal.fit(X_train_bal, y_train_bal)
    models['LogisticRegression'] = (lr, lr_bal)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_bal = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_bal.fit(X_train_bal, y_train_bal)
    models['RandomForest'] = (rf, rf_bal)
    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_bal = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_bal.fit(X_train_bal, y_train_bal)
    models['XGBoost'] = (xgb, xgb_bal)
    # Evaluate
    for name, (model, model_bal) in models.items():
        results[name] = {
            'Imbalanced': model.predict(X_test),
            'Balanced': model_bal.predict(X_test)
        }
    return models, results
