import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_movielens(data_path='data/ml-100k/u.data', test_size=0.2, val_size=0.1):
    # Load the dataset
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(data_path, sep='\t', names=column_names)
    data['rating'] = data['rating'] / data['rating'].max()
    
    # Split the data into train and temp datasets
    train_data, temp_data = train_test_split(data, test_size=test_size + val_size, random_state=42)
    
    # Calculate the proportion of validation data in the temp dataset
    val_proportion = val_size / (test_size + val_size)
    
    # Split the temp dataset into validation and test datasets
    val_data, test_data = train_test_split(temp_data, test_size=val_proportion, random_state=42)
    
    return train_data, val_data, test_data

# Example usage:
# train_data, val_data, test_data = load_and_split_movielens()

if __name__ == "__main__":
    train_data, val_data, test_data = load_and_split_movielens()
    print(train_data.head())
    print(val_data.head())
    print(test_data.head())
