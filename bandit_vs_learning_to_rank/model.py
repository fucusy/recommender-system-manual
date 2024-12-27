import torch
import torch.nn as nn

class PointWiseModel(nn.Module):
    def __init__(self, num_titles):
        self.embedding_dim = 16
        self.hidden_dim = 8
        super(PointWiseModel, self).__init__()

        self.embedding_layer = nn.Embedding(num_titles, self.embedding_dim)
        self.linear1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, title_ids):
        # Get embeddings for title ids
        embeddings = self.embedding_layer(title_ids)
        # Forward pass through the network
        h = torch.relu(self.linear1(embeddings))
        score = self.linear2(h)
        batch_size = title_ids.shape[0]
        return score.view(batch_size)
    

class PairWiseModel(nn.Module):
    def __init__(self, num_titles):
        self.embedding_dim = 16
        self.hidden_dim = 8
        super(PairWiseModel, self).__init__()

        self.embedding_layer = nn.Embedding(num_titles, self.embedding_dim)
        self.linear1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, title_ids):
        # Get embeddings for title ids
        embeddings = self.embedding_layer(title_ids)
        # Forward pass through the network
        h = torch.relu(self.linear1(embeddings))
        score = self.linear2(h)
        return score