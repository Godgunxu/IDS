# main.py
import argparse
import train
import test

def train_model(train_path):
    print("Training model...")
    train.train_model(train_path)  # Pass the train_path to the train module's function

def test_model(test_path, model_path):
    print("Testing model...")
    test.test_model(test_path, model_path)  # Pass the test_path and model_path to the test module's function

def main():
    parser = argparse.ArgumentParser(description='Train or test a decision tree model.')
    parser.add_argument('--option', choices=['train', 'test'], default='test', help='Specify whether to train or test the model.')
    parser.add_argument('--train_path', default="./flows_benign_and_DoS.csv", help='Path to the training dataset.')
    parser.add_argument('--test_path', default="../test.csv", help='Path to the test dataset.')
    parser.add_argument('--model_path', default="./decision_tree_model.pkl", help='Path to the model file for testing.')
    args = parser.parse_args()

    if args.option == 'train':
        train_model(args.train_path)
    elif args.option == 'test':
        test_model(args.test_path, args.model_path)

if __name__ == "__main__":
    main()
