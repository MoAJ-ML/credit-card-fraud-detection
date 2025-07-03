from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_models(models, results, X_test, y_test):
    """
    Prints metrics and plots confusion matrices, ROC and PR curves for all models.
    """
    for name, (model, model_bal) in models.items():
        print(f"\n=== {name} (Imbalanced) ===")
        _evaluate_single(model, X_test, y_test, name+" (Imbalanced)")
        print(f"\n=== {name} (Balanced/SMOTE) ===")
        _evaluate_single(model_bal, X_test, y_test, name+" (Balanced)")
        if hasattr(model_bal, 'feature_importances_'):
            _plot_feature_importance(model_bal, X_test, name)

def _evaluate_single(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # ROC
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0,1],[0,1],'k--')
        plt.title(f'ROC Curve: {label}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()
        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
        plt.title(f'Precision-Recall Curve: {label}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

def _plot_feature_importance(model, X_test, name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]
    features = X_test.columns[indices]
    plt.figure(figsize=(8,4))
    sns.barplot(x=importances[indices], y=features)
    plt.title(f'Feature Importances ({name})')
    plt.show()

def summarize_results(results):
    print("\nSummary: Compare F1, Precision, Recall, ROC-AUC, PR-AUC for all models.\n")
    # This can be extended for tabular summary if needed.
