Here’s the complete .md content for your README file:

# Network Flow Classification Using Decision Trees

**Authors:** Y. Chen & G. Xu  
**Date:** April 10, 2024  

This project focuses on classifying network flow data into various categories using Decision Trees. It involves training and testing a Decision Tree model on network flow datasets and evaluating its performance using key metrics.

---

## Project Structure

- **`Cicflowmeter`**: Processes `.pcap` files into `.csv` format.
- **`Dataset_kali`**: Contains the dataset and the trained model specific to our dataset.
- **`Dos_attack_code`**: Source code for generating DoS attack traffic.
- **`image`**: Contains test result visualizations.
- **`main.py`**: Main script for training and testing the Decision Tree model.
- **`train.py`**: Script for training the Decision Tree model.
- **`test.py`**: Script for testing the trained Decision Tree model.
- **`decision_tree_model.pkl`**: Pre-trained Decision Tree model on `flows_benign_and_DoS.csv`.
- **`model_our_dataset.pkl`**: Pre-trained Decision Tree model on `dataset_kali.csv`.
- **`flows_benign_and_DoS.csv`**: CSV file containing the dataset for training the model.
- **`test.csv`**: CSV file containing the dataset for testing the model.
- **`README.md`**: Provides an overview of the project.

---

## Dependencies

Ensure the following dependencies are installed:

- Python 3.x  
- NumPy  
- scikit-learn  
- matplotlib  

Install them using pip:  
```bash
pip install numpy scikit-learn matplotlib

Usage

Training the Model

To train the Decision Tree model, use the following command:

python main.py --option train --train_path ./flows_benign_and_DoS.csv

Replace ./flows_benign_and_DoS.csv with the path to your dataset for training.

Testing the Model

To test the Decision Tree model, use the following command:

python main.py --option test --test_path ../test.csv --model_path ./decision_tree_model.pkl

Replace ../test.csv with the path to your test dataset.

Testing the Model on Our Dataset

To test the model on our custom dataset, use this command:

python main.py --option test --test_path ./Dataset_kali/dataset.csv --model_path ./Dataset_kali/our_model.pkl

Test Script Example

To compute and print evaluation metrics, replace the dataset path and run:

python main.py --option test --test_path ../test.csv --model_path ./decision_tree_model.pkl

Or simply use:

python test.py

Project Highlights
	•	Flexible Usage: Easily adaptable for other datasets.
	•	Custom Dataset Support: Includes pre-trained models for specific datasets.
	•	Key Tools: Utilizes scikit-learn for Decision Tree implementation.

Contributing

We welcome contributions! Whether you have feature suggestions or bug reports, please submit an issue or pull request to this repository.

This README provides clear instructions for setting up, running, and contributing to the project. If you have questions or need support, feel free to reach out!

