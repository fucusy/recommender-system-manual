import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from load_data import load_and_split_movielens
import pickle

class RecommenderNN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super(RecommenderNN, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

    def forward(self, user_id, item_id):
        user_embedded = self.user_embedding(user_id)
        item_embedded = self.item_embedding(item_id)
        return torch.cosine_similarity(user_embedded, item_embedded)

def train_model(model, train_data, val_data, n_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        user_ids = torch.tensor(train_data['user_id'].values, dtype=torch.long)
        item_ids = torch.tensor(train_data['item_id'].values, dtype=torch.long)
        ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)
        
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_user_ids = torch.tensor(val_data['user_id'].values, dtype=torch.long)
            val_item_ids = torch.tensor(val_data['item_id'].values, dtype=torch.long)
            val_ratings = torch.tensor(val_data['rating'].values, dtype=torch.float32)
            
            val_predictions = model(val_user_ids, val_item_ids)
            val_loss = criterion(val_predictions, val_ratings)
            print(f'Validation Loss: {val_loss.item():.4f}')

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        test_user_ids = torch.tensor(test_data['user_id'].values, dtype=torch.long)
        test_item_ids = torch.tensor(test_data['item_id'].values, dtype=torch.long)
        test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32)
        
        test_predictions = model(test_user_ids, test_item_ids)
        mse = mean_squared_error(test_ratings.numpy(), test_predictions.numpy())
        print(f'Test MSE: {mse:.4f}')

if __name__ == "__main__":
    # Load the data
    train_data, val_data, test_data = load_and_split_movielens()
    print(f'Training data size: {len(train_data)}')
    print(f'Validation data size: {len(val_data)}')
    print(f'Test data size: {len(test_data)}')
    print(f'Training data statistics:\n{train_data.describe()}')
    print(f'Validation data statistics:\n{val_data.describe()}')
    print(f'Test data statistics:\n{test_data.describe()}')
    
    n_users = max(train_data['user_id'].max(), val_data['user_id'].max(), test_data['user_id'].max()) + 1
    n_items = max(train_data['item_id'].max(), val_data['item_id'].max(), test_data['item_id'].max()) + 1
    embedding_dim = 10
    
    # Initialize the model
    model = RecommenderNN(n_users, n_items, embedding_dim)
    
    # Train the model
    train_model(model, train_data, val_data, n_epochs=1000, lr=0.01)
    
    # Evaluate the model
    evaluate_model(model, test_data)

    import os
    import torch

    # Create the models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model
    model_path = 'models/recommender_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # Save the item factors
    item_factors = model.item_embedding.weight.data.numpy()
    item_factors_dict = {i: item_factors[i] for i in range(len(item_factors))}
    item_factors_path = 'models/item_factors_nn.pkl'
    with open(item_factors_path, 'wb') as f:
        pickle.dump(item_factors_dict, f)
    print(f'Item factors saved to {item_factors_path}')