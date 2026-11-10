from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def evaluate_models(models, results, X_test, y_test):
    """
    Prints metrics and plots confusion matrices, ROC and PR curves for all models.
    """
    metrics = {}
    for name, (model, model_bal) in models.items():
        print(f"\n=== {name} (Imbalanced) ===")
        metrics[name + ' (Imbalanced)'] = _evaluate_single(model, X_test, y_test, name+" (Imbalanced)")
        print(f"\n=== {name} (Balanced/SMOTE) ===")
        metrics[name + ' (Balanced)'] = _evaluate_single(model_bal, X_test, y_test, name+" (Balanced)")
        if hasattr(model_bal, 'feature_importances_'):
            _plot_feature_importance(model_bal, X_test, name)
    return metrics

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
        pr_auc = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f'Average Precision = {pr_auc:.4f}')
        plt.title(f'Precision-Recall Curve: {label}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
    return {
        'threshold': 0.5,
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
        'average_precision': float(average_precision_score(y_test, y_prob)) if y_prob is not None else None,
        'confusion_matrix': cm.tolist(),
    }

def _plot_feature_importance(model, X_test, name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]
    features = X_test.columns[indices]
    plt.figure(figsize=(8,4))
    sns.barplot(x=importances[indices], y=features)
    plt.title(f'Feature Importances ({name})')
    plt.show()

def summarize_results(results):
    with open('validated_metrics.json', 'w') as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))
