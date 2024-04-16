# main.py

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

def train_model(train_path):
    # Load the data from the CSV file
    print(f"Training model with data from: {train_path}")
    with open(train_path, newline='') as f:
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
    selected_columns = [8, 9, 10, 24, 45, 83] #  Flow Duration, Total Fwd Packets, Total Backward Packets,  Idle Min
    X = data_arr[:, selected_columns].astype(np.float64)
    y = target_arr

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=42)

    # Initialize and train the Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'decision_tree_model.pkl')
    print("Model saved successfully!")

    # Evaluate the model
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    print(f'Training Accuracy: {score_train}')
    print(f'Testing Accuracy: {score_test}')

    # Plot the decision tree
    plt.figure(figsize=(30, 15))
    feature_names_list = data[0][:-1]
    plot_tree(model, filled=True, feature_names=feature_names_list, class_names=target_names)
    plt.title('Decision Tree for Network Flow Classification')
    plt.show()

    # Compute ROC curve and ROC area for each class
    y_scores = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    epsilon = 1e-10  # Small value to prevent division by zero
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
    predictions = model.predict(X_test)
    fpr = dict()
    dr = dict()
    for i in range(len(target_names)):
        false_positives = np.sum((predictions == i) & (y_test != i))  # Predicted class i, actual not i
        true_positives = np.sum((predictions == i) & (y_test == i))   # Predicted class i, actual i
        false_negatives = np.sum((predictions != i) & (y_test == i))  # Predicted not i, actual i
        true_negatives = np.sum((predictions != i) & (y_test != i))   # Predicted not i, actual not i

        # False Positive Rate (FPR) = FP / (FP + TN)
        fpr[i] = false_positives / (false_positives + true_negatives+epsilon)

        # Detection Rate (DR) = TP / (TP + FN)
        dr[i] = true_positives / (true_positives + false_negatives+epsilon)

        print(f'Detection Rate (DR) for class {target_names[i]}: {dr[i]}')
        print(f'False Positive Rate (FPR) for class {target_names[i]}: {fpr[i]}')


if __name__ == "__main__":
    train_model("./flows_benign_and_DoS.csv")
