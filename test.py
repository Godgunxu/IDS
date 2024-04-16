# main.py

import csv
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def test_model(test_path, model_path):
    # Load test data from the CSV file
    with open(test_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Extract the target names and map them to numerical values
    target_names = ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris']
    class_mapping = {name: index for index, name in enumerate(target_names)}

    # Convert the target column to numerical form using the mapping
    target = [class_mapping[row[-1]] for row in data[1:]]  # Exclude the header row
    target_arr = np.array(target)

    # Convert the data to a NumPy array and exclude the target column
    data_arr = np.array([row[:-1] for row in data[1:]])

    # Select columns for the analysis
    selected_columns = [8, 9, 10, 24, 45, 83]
    X_test = data_arr[:, selected_columns].astype(np.float64)
    y_test = target_arr

    # Load the saved model
    loaded_model = joblib.load(model_path)

    # Predict probabilities for each class
    y_scores = loaded_model.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    lw = 2
    for i in range(len(target_names)):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {} (area = {:0.2f})'.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Calculate False Positive Rate (FPR) and Detection Rate (DR)
    predictions = loaded_model.predict(X_test)
    fpr = dict()
    dr = dict()
    for i in range(len(target_names)):
        false_positives = np.sum((predictions == i) & (y_test != i))  # Predicted class i, actual not i
        true_positives = np.sum((predictions == i) & (y_test == i))   # Predicted class i, actual i
        false_negatives = np.sum((predictions != i) & (y_test == i))  # Predicted not i, actual i
        true_negatives = np.sum((predictions != i) & (y_test != i))   # Predicted not i, actual not i

        # False Positive Rate (FPR) = FP / (FP + TN)
        fpr[i] = false_positives / (false_positives + true_negatives)

        # Detection Rate (DR) = TP / (TP + FN)
        dr[i] = true_positives / (true_positives + false_negatives)

        print(f'Detection Rate (DR) for class {target_names[i]}: {dr[i]}')
        print(f'False Positive Rate (FPR) for class {target_names[i]}: {fpr[i]}')


if __name__ == "__main__":
    test_model(test_path="../test.csv", model_path="./decision_tree_model.pkl")
