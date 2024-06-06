Network Flow Classification using Decision Trees
Author: Y.Chen & G.Xu
Date: April 10, 2024

This project aims to classify network flow data into different categories using Decision Trees. It involves training a Decision Tree model on network flow data and evaluating its performance using various metrics.
---

#  Test Script: to compute and print the evaluation metrics,replace "../test.csv" with your dataset and run: 

```
python main.py --option test --test_path ../test.csv --model_path ./decision_tree_model.pkl
```
# or 

```
python test.py
```


# Network Flow Classification using Decision Trees

This project aims to classify network flow data into different categories using Decision Trees. It involves training a Decision Tree model on network flow data and evaluating its performance using various metrics.

## Project Structure

- `Cicflowmeter`: process the pcap file into a csv file.
- `Dataset_kali`: contain our obtained dataset and model on our dataset.
- `Dos_attack_code`: contain sources codes of we use to launch DoS attack.
- `image`: contain test result.
- `main.py`: The main script to run the project. It allows training and testing the Decision Tree model.
- `train.py`: Contains the code for training the Decision Tree model.
- `test.py`: Contains the code for testing the trained Decision Tree model.
- `decision_tree_model.pkl`: The trained Decision Tree model on "flows_benign_and_Dos.csv" saved as a pickle file.
- `model_our_dataset.pkl`: The trained Decision Tree model on "dataset_kali.csv" saved as a pickle file.
- `flows_benign_and_DoS.csv`: CSV file containing the network flow data used for training the model.
- `test.csv`: CSV file containing the network flow data used for testing the model.
- `README.md`: This file providing an overview of the project.


## Usage

### Dependencies
- Python 3.x
- NumPy
- scikit-learn
- matplotlib

Install the dependencies using pip:
```
pip install numpy scikit-learn matplotlib
```

### Training the Model
To train the Decision Tree model, execute the following command:
```
python main.py --option train --train_path ./flows_benign_and_DoS.csv
```
Replace `./flows_benign_and_DoS.csv` with the path to your training dataset.

### Testing the Model
***To run test script, execute the following command:***
```
python main.py --option test --test_path ../test.csv --model_path ./decision_tree_model.pkl
```
Replace `../test.csv` with the path to your test dataset.

### Testing the Model on our dataset
```
python main.py --option test --test_path ./Dataset_kali/dataset.csv --model_path ./Dataset_kali/our_model.pkl
```

# Contributing
Contributions to this project are welcome! Whether it's feature suggestions or bug reports, please feel free to submit an issue or pull request.
