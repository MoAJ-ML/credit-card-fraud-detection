import os
from src import data_exploration, preprocessing, modeling, evaluation

def main():
    # Step 1: Data Loading & Exploration
    df = data_exploration.load_and_explore_data()
    
    # Step 2: Data Preprocessing
    X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = preprocessing.preprocess_data(df)

    # Step 3: Model Building
    models, results = modeling.train_and_compare_models(X_train, X_test, y_train, y_test, X_train_bal, y_train_bal)

    # Step 4: Model Evaluation
    metrics = evaluation.evaluate_models(models, results, X_test, y_test)

    # Step 5: Results & Insights
    evaluation.summarize_results(metrics)

if __name__ == '__main__':
    main()
