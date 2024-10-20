import numpy as np
from load_data import load_and_split_movielens
from sklearn.metrics import mean_squared_error


# ALS algorithm implementation
class ALS:
    def __init__(self, n_factors=10, n_iterations=10, lambda_reg=0.1):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg

    def fit(self, ratings):
        self.user_ids = ratings['user_id'].unique()
        self.item_ids = ratings['item_id'].unique()
        self.user_map = {user_id: i for i, user_id in enumerate(self.user_ids)}
        self.item_map = {item_id: i for i, item_id in enumerate(self.item_ids)}            
        self.user_idx_to_id = {i: user_id for i, user_id in enumerate(self.user_ids)}
        self.item_idx_to_id = {i: item_id for i, item_id in enumerate(self.item_ids)}
        
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)
        
        self.user_factors = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors))
        
        for _ in range(self.n_iterations):
            self.user_factors = self._als_step(ratings, self.user_factors, self.item_factors, self.user_map, self.item_map, self.n_users, self.n_items, self.lambda_reg, True)
            self.item_factors = self._als_step(ratings, self.user_factors, self.item_factors, self.user_map, self.item_map, self.n_users, self.n_items, self.lambda_reg, False)
            # Calculate and print the loss for this iteration
            mse = self._calculate_mse(ratings)
            print(f"Iteration {_ + 1}/{self.n_iterations}, MSE: {mse:.4f}")

    def _calculate_mse(self, ratings):
        predicted = np.zeros(len(ratings))
        for i, (_, row) in enumerate(ratings.iterrows()):
            predicted[i] = self.predict(row['user_id'], row['item_id'])
        return mean_squared_error(ratings['rating'], predicted)

    
    def _als_step(self, ratings, user_factors, item_factors, user_map, item_map, n_users, n_items, lambda_reg, user_step):
        if user_step:
            YTY = item_factors.T.dot(item_factors)
            lambdaI = np.eye(YTY.shape[0]) * lambda_reg
            for u in range(n_users):
                user_ratings = ratings[ratings['user_id'] == self.user_ids[u]]
                if len(user_ratings) == 0:
                    continue
                item_ids = [item_map[item_id] for item_id in user_ratings['item_id']]
                Y = item_factors[item_ids]
                A = Y.T.dot(Y) + lambdaI
                b = Y.T.dot(user_ratings['rating'])
                user_factors[u] = np.linalg.solve(A, b)
        else:
            XTX = user_factors.T.dot(user_factors)
            lambdaI = np.eye(XTX.shape[0]) * lambda_reg
            for i in range(n_items):
                item_ratings = ratings[ratings['item_id'] == self.item_ids[i]]
                if len(item_ratings) == 0:
                    continue
                user_ids = [user_map[user_id] for user_id in item_ratings['user_id']]
                X = user_factors[user_ids]
                A = X.T.dot(X) + lambdaI
                b = X.T.dot(item_ratings['rating'])
                item_factors[i] = np.linalg.solve(A, b)
        return user_factors if user_step else item_factors

    def predict(self, user_id, item_id):
        if user_id in self.user_map and item_id in self.item_map:
            user_idx = self.user_map[user_id]
            item_idx = self.item_map[item_id]
            return self.user_factors[user_idx, :].dot(self.item_factors[item_idx, :])
        else:
            return np.nan

    def get_item_factors(self):
        return {self.item_idx_to_id[i]: self.item_factors[i] for i in range(len(self.item_ids))}

if __name__ == "__main__":
    # Load the data
    train_data, val_data, test_data = load_and_split_movielens()

    # Train the ALS model
    als = ALS(n_factors=10, n_iterations=10, lambda_reg=0.1)
    als.fit(train_data)

    # Predict ratings for validation data
    val_data['predicted_rating'] = val_data.apply(lambda row: als.predict(row['user_id'], row['item_id']), axis=1)

    # Calculate mean squared error, excluding NaN values for validation data
    valid_predictions = val_data.dropna(subset=['predicted_rating'])
    nan_count = len(val_data) - len(valid_predictions)
    
    mse = mean_squared_error(valid_predictions['rating'], valid_predictions['predicted_rating'])
    print(f'Validation MSE: {mse:.4f}')
    print(f'Number of NaN predictions in validation data: {nan_count}')
    print(f'Size of validation data: {len(val_data)}')

    # Predict ratings for test data
    test_data['predicted_rating'] = test_data.apply(lambda row: als.predict(row['user_id'], row['item_id']), axis=1)

    # Calculate mean squared error, excluding NaN values for test data
    test_predictions = test_data.dropna(subset=['predicted_rating'])
    nan_count_test = len(test_data) - len(test_predictions)
    
    mse_test = mean_squared_error(test_predictions['rating'], test_predictions['predicted_rating'])
    print(f'Test MSE: {mse_test:.4f}')
    print(f'Number of NaN predictions in test data: {nan_count_test}')
    print(f'Size of test data: {len(test_data)}')

    import os
    import pickle

    # Create the models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the ALS factors for users and items
    user_factors_path = 'models/user_factors.pkl'
    item_factors_path = 'models/item_factors.pkl'

    with open(user_factors_path, 'wb') as f:
        pickle.dump(als.user_factors, f)
    print(f'User factors saved to {user_factors_path}')

    with open(item_factors_path, 'wb') as f:
        pickle.dump(als.get_item_factors(), f)
    print(f'Item factors saved to {item_factors_path}')



