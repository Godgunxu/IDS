# Network Flow Classification Using Decision Trees

**Authors:** G. Xu  
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

## Contributions
We welcome contributions! Whether you have feature suggestions or bug reports, please submit an issue or pull request to this repository.

