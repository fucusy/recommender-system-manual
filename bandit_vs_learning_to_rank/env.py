import numpy as np
import torch
import torch.nn as nn
from loss import PairwiseHingeLoss

class RecommenderEnv:
    
    def __init__(self, num_titles, click_probabilities=None, exit_probability=0.5, random_seed=None):
        """Initialize recommender environment.
        
        Args:
            num_titles (int): Number of titles/items in the system
            click_probabilities (list/array, optional): Click probability for each title. 
                If None, randomly generated between 0-1. Length must match num_titles.
            exit_probability (float, optional): Probability of user exiting the app after checking a title.
                Must be between 0 and 1. Default is 0.0.
            random_seed (int, optional): Random seed for reproducibility
        """
        self.num_titles = num_titles
        self.exit_probability = exit_probability
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if click_probabilities is None:
            # Generate random click probabilities between 0 and 0.1
            self.click_probabilities = np.random.random(num_titles) * 0.1
        else:
            if len(click_probabilities) != num_titles:
                raise ValueError("Length of click_probabilities must match num_titles")
            self.click_probabilities = np.array(click_probabilities)
            
        # Validate probabilities are between 0 and 1
        if not np.all((self.click_probabilities >= 0) & (self.click_probabilities <= 1)):
            raise ValueError("All click probabilities must be between 0 and 1")
        
        if not (0 <= self.exit_probability <= 1):
            raise ValueError("Exit probability must be between 0 and 1")

    def get_title_list(self):
        return list(range(self.num_titles))

    def get_click(self, title_id):
        """Get click outcome for a shown title based on its probability.
        
        Args:
            title_id (int): ID of the title shown
            
        Returns:
            int: 1 if clicked, 0 if not clicked
        """
        if not 0 <= title_id < self.num_titles:
            raise ValueError(f"Title ID must be between 0 and {self.num_titles-1}")
            
        return int(np.random.random() < self.click_probabilities[title_id])


    def get_user_feedback(self, title_rankings):
        """Get user feedback for a list of title rankings.
        
        Args:
            title_rankings (list/array): List of title rankings
            
        Returns:
            array of the same length as title_rankings with CLICK, SKIP, or NOT_SEEN
        """
        feedback = []
        for title_id in title_rankings:
            if np.random.random() < self.click_probabilities[title_id]:
                feedback.append("CLICK")                
            else:
                feedback.append("SKIP")
            if np.random.random() < self.exit_probability:
                break
        feedback.extend(["NOT_SEEN"] * (len(title_rankings) - len(feedback)))
        return feedback
    
    def get_maximum_expected_reward(self):
        # sort the click probabilities in descending order
        sorted_click_probabilities = sorted(enumerate(self.click_probabilities), key=lambda x: x[1], reverse=True)

        # loop through the sorted probabilities and calculate the expected reward
        expected_reward = 0
        seen_probability = 1
        for title_id, click_probability in sorted_click_probabilities:
            expected_reward += click_probability * seen_probability
            seen_probability *= (1 - self.exit_probability)

        return expected_reward    

    def manual_edit_ranking(self):
        """
        Suppose the editor has knowledge of some titles, the selected top x% titles, half are actualy top x%, and other half are random titles
        """

        sorted_click_probabilities = sorted(enumerate(self.click_probabilities), key=lambda x: x[1], reverse=True)
        top_percent = int(self.num_titles * 0.2)
        top_percent_titles = sorted_click_probabilities[:top_percent]

        # set seed for reproducibility and stability in different days
        rng = np.random.default_rng(seed=1)
        half_good_selected_titles = rng.choice(top_percent_titles, size=int(top_percent * 0.5), replace=False)
        half_good_selected_titles_ids = [int(title_id) for title_id, _ in half_good_selected_titles]

        remaining_titles = [int(title_id) for title_id, _ in sorted_click_probabilities if title_id not in half_good_selected_titles_ids]
        half_random_selected_titles_ids = rng.choice(remaining_titles, size=int(top_percent * 0.5), replace=False)
        
        selected_titles_ids = np.concatenate((half_good_selected_titles_ids, half_random_selected_titles_ids))
        rng.shuffle(selected_titles_ids)

        # sort the selected titles by their click probabilities
        other_title_ids = [int(title_id) for title_id, _ in sorted_click_probabilities if title_id not in selected_titles_ids]
        rng.shuffle(other_title_ids)
        return np.concatenate((selected_titles_ids, other_title_ids))

class Agent:
    def __init__(self, num_titles):
        self.num_titles = num_titles

    def make_ranking(self):
        raise NotImplementedError("Subclasses must implement the make_ranking method")

    def understand_agent(self):
        raise NotImplementedError("Subclasses must implement the understand_agent method")

    def collect_user_feedback(self, title_ranking, user_feedback):
        raise NotImplementedError("Subclasses must implement the collect_user_feedback method")
    
    def day_starts(self):
        """
        This method is called at the end of each day. 
        The agent can use this method to train their model if the model is designed to updated daily.
        """
        raise NotImplementedError("Subclasses must implement the day_starts method")

class BanditAgent(Agent):
    """
    Bandit agent using Thompson Sampling
    """
    def __init__(self, num_titles):
        self.epsilon = 0.1
        self.num_titles = num_titles
        self.click_counts = np.zeros(num_titles)
        self.seen_counts = np.zeros(num_titles)
   
    def collect_user_feedback(self, title_ranking, user_feedback):
        self.update_theta(title_ranking, user_feedback)

    def make_ranking(self):
        if np.random.rand() < self.epsilon:
            return np.random.permutation(self.num_titles)
        else:
            return self.make_exploitation_ranking()

    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("click_counts: ", self.click_counts)
        print("seen_counts: ", self.seen_counts)
        click_probabilities = self.click_counts / self.seen_counts
        print("click_probabilities: ", click_probabilities)

    def day_starts(self):
        pass

    def update_theta(self, title_ids, user_feedback):
        """
        :param title_ids: list of title ids that were shown to the user, e.g. [0, 1, 2, ...]
        :param user_feedback: list of feedback from user, e.g. ["CLICK", "SKIP", "NOT_SEEN", ...]
        Update theta based on user feedback
        """
        for title_id, feedback in zip(title_ids, user_feedback):
            if feedback == "CLICK":
                self.click_counts[title_id] += 1
                self.seen_counts[title_id] += 1
            elif feedback == "SKIP":
                self.seen_counts[title_id] += 1

    def make_exploitation_ranking(self):
        #TODO if the score is the same, randomize the order
        expected_rewards = []
        for click_count, seen_count in zip(self.click_counts, self.seen_counts):
            # Use Thompson sampling to sample a number based on seen count and click count of a title
            #TODO verify this formula
            sampled_value = np.random.beta(click_count + 1, seen_count - click_count + 1) 
            expected_rewards.append(sampled_value)
        
        # Order title_id based on the sampled probabilities
        title_ranking = np.argsort(expected_rewards)[::-1]
        return title_ranking



class BucketSortingAgent(Agent):
    """
    Bucket Sorting Agent
    """
    def __init__(self, num_titles):
        self.bucket_size = min(max(int(num_titles / 100), 1), 5)
        self.num_titles = num_titles
        self.click_counts = np.zeros(num_titles)
        self.seen_counts = np.zeros(num_titles)
        self.click_through_rates = np.zeros(num_titles)
   
    def collect_user_feedback(self, title_ranking, user_feedback):
        self.update_parameters(title_ranking, user_feedback)

    def make_ranking(self):
        # sort by click count first, group them into 10 buckets, the clicks in each bucket are equal, means the number of clicks in each bucket is 10% of the total clicks
        sorted_click_counts = sorted(enumerate(self.click_counts), key=lambda x: x[1], reverse=True)
        buckets = [[] for _ in range(10)]
        total_click_count = sum(self.click_counts)
        accumulated_click_count = 0
        bucket_clicks_quota = total_click_count / self.bucket_size 
        for title_id, click_count in sorted_click_counts:
            accumulated_click_count += click_count
            bucket_index = min(int(accumulated_click_count / bucket_clicks_quota), self.bucket_size - 1)
            buckets[bucket_index].append(title_id)
        
        # sort each bucket by click through rate
        for bucket in buckets:
            bucket.sort(key=lambda x: self.click_through_rates[x], reverse=True)
        
        # flatten the buckets
        title_ranking = [title_id for bucket in buckets for title_id in bucket]
        return title_ranking

    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("bucket_size: ", self.bucket_size)
        print("click_counts: ", self.click_counts)
        print("click_through_rates: ", self.click_through_rates)

    def day_starts(self):
        pass

    def update_parameters(self, title_ids, user_feedback):
        """
        :param title_ids: list of title ids that were shown to the user, e.g. [0, 1, 2, ...]
        :param user_feedback: list of feedback from user, e.g. ["CLICK", "SKIP", "NOT_SEEN", ...]
        Update parameters based on user feedback
        """
        for title_id, feedback in zip(title_ids, user_feedback):
            if feedback != "NOT_SEEN":
                if feedback == "CLICK":
                    self.click_counts[title_id] += 1
                    self.seen_counts[title_id] += 1
                elif feedback == "SKIP":
                    self.seen_counts[title_id] += 1
                self.click_through_rates[title_id] = self.click_counts[title_id] / self.seen_counts[title_id]
        

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

class PointWiseModelAgent(Agent):
    def __init__(self, num_titles):
        self.num_titles = num_titles        
        self.model = PointWiseModel(num_titles)
        self.collected_training_data = [] # each element is a tuple of (title_ids, user_feedback)
        self.valid_title_ids = [] # each element is a title id
        self.valid_user_feedback = [] # each element is a user feedback, 1 for CLICK, 0 for SKIP

    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback):
        for title_id, feedback in zip(title_ids, user_feedback):
            if feedback != "NOT_SEEN":
                self.valid_title_ids.append(title_id)
                if feedback == "CLICK":
                    self.valid_user_feedback.append(1)
                else:
                    self.valid_user_feedback.append(0)
            else:
                break
        self.accumulate_user_feedback(title_ids, user_feedback)
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.collected_training_data) > 0:
            self.train()

    def accumulate_user_feedback(self, title_ids, user_feedback):
        self.collected_training_data.append((title_ids, user_feedback))

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = nn.BCEWithLogitsLoss()

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, valid_title_ids, valid_user_feedback):
                self.valid_title_ids = valid_title_ids
                self.valid_user_feedback = valid_user_feedback

            def __len__(self):
                return len(self.valid_title_ids)

            def __getitem__(self, idx):
                title_ids = self.valid_title_ids[idx]
                user_feedback = self.valid_user_feedback[idx]
                # Convert "SKIP" to 0 and "CLICK" to 1
                return torch.tensor(title_ids, dtype=torch.long), torch.tensor(user_feedback, dtype=torch.float32)

        dataset = UserFeedbackDataset(self.valid_title_ids, self.valid_user_feedback)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(100):
            epoch_loss = 0.0
            data_size = 0
            for title_ids, user_feedback in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_ids)
                loss = criterion(outputs, user_feedback)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_ids)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")


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


class PairWiseModelAgent(Agent):
    def __init__(self, num_titles):
        self.num_titles = num_titles        
        self.model = PairWiseModel(num_titles)
        self.collected_training_data = [] # each element is a tuple of (title_ids, user_feedback)
        self.clean_training_data = [] # each element is a tuple of (title_ids, list of 1 or 0)

    def make_ranking(self):
        # Get scores for all titles
        title_ids = torch.arange(self.num_titles)
        scores = self.model(title_ids)
        # Return titles sorted by their scores (highest first)
        title_scores = list(zip(title_ids, scores))
        title_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_title_ids = [title_id for title_id, score in title_scores]
        return torch.tensor(sorted_title_ids)

    def collect_user_feedback(self, title_ids, user_feedback):
        clean_user_feedback = []
        clean_title_ids = []
        click_count = 0
        for title_id, feedback in zip(title_ids, user_feedback):
            if feedback != "NOT_SEEN":
                clean_user_feedback.append(1 if feedback == "CLICK" else 0)
                clean_title_ids.append(title_id)
                if feedback == "CLICK":
                    click_count += 1
            else:
                break
        if len(clean_title_ids) > 1 and click_count > 0:
            self.clean_training_data.append((clean_title_ids, clean_user_feedback))
        self.accumulate_user_feedback(title_ids, user_feedback)
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.collected_training_data) > 0:
            self.train()

    def accumulate_user_feedback(self, title_ids, user_feedback):
        self.collected_training_data.append((title_ids, user_feedback))

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = PairwiseHingeLoss()

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, clean_training_data):
                self.clean_training_data = clean_training_data

            def __len__(self):
                return len(self.clean_training_data)

            def __getitem__(self, idx):
                title_ids, user_feedback = self.clean_training_data[idx]
                return torch.tensor(title_ids, dtype=torch.long), torch.tensor(user_feedback, dtype=torch.float32), torch.tensor(len(title_ids), dtype=torch.long)

        dataset = UserFeedbackDataset(self.clean_training_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for epoch in range(100):
            epoch_loss = 0.0
            data_size = 0
            for title_ids, user_feedback, n in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_ids)[0].reshape(1, -1)
                user_feedback = user_feedback.reshape(1, -1)
                loss = criterion(outputs, user_feedback, n).sum()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_ids)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")

if __name__ == "__main__":
    title_size = 1000
    user_size = 10 * title_size
    run_days = 10

    # if True, the manual edit stage is included on the first few days, and the collected data those days are avaiable to the agent
    include_manual_edit_stage = True 
    edit_stage_days = 3

    total_days = run_days + edit_stage_days if include_manual_edit_stage else run_days 
    
    import time
    start_time = time.time()
    env = RecommenderEnv(num_titles=title_size)
    bandit = BanditAgent(num_titles=title_size)
    pointwise_agent = PointWiseModelAgent(num_titles=title_size)
    pairwise_agent = PairWiseModelAgent(num_titles=title_size)
    bucket_agent = BucketSortingAgent(num_titles=title_size)

    agents = [bandit, bucket_agent, pointwise_agent, pairwise_agent]

    # sort by probability, print the title id and value
    sorted_probabilities = sorted(enumerate(env.click_probabilities), key=lambda x: x[1], reverse=True)
    for title_id, probability in sorted_probabilities:
        print(f"title {title_id}: {round(probability, 2)}")

    maximum_expected_reward = env.get_maximum_expected_reward()
    
    for agent in agents:
        actual_reward = 0.0
        max_reward = 0.0
        actual_reward_list = []
        max_reward_list = []
        print("agent name: ", agent.__class__.__name__)
        for day in range(total_days):
            actual_reward_the_day = 0.0
            max_reward_the_day = 0.0
            manual_edit_stage = include_manual_edit_stage and day < edit_stage_days
            if not manual_edit_stage:
                agent.day_starts()
                
            for user_idx in range(user_size):
                if user_idx % 1000 == 0:
                    print(f"user {user_idx} / {user_size}")

                if manual_edit_stage:
                    title_ranking = env.manual_edit_ranking()
                else:
                    title_ranking = agent.make_ranking()

                user_feedback = env.get_user_feedback(title_ranking)
                
                click_count = sum(1 for feedback in user_feedback if feedback == "CLICK")
                actual_reward_the_day += click_count
                max_reward_the_day += maximum_expected_reward
                agent.collect_user_feedback(title_ranking, user_feedback)
                
            actual_reward += actual_reward_the_day
            max_reward += max_reward_the_day
            actual_reward_list.append(actual_reward)
            max_reward_list.append(max_reward)
            
            print(f"Day {day}, actual_reward: {actual_reward_the_day}, max_reward: {max_reward_the_day}")
        agent.understand_agent()
        print("actual_reward (rounded): ", round(actual_reward, 2))
        print("max_reward (rounded): ", round(max_reward, 2))
    # import matplotlib.pyplot as plt
    # plt.plot(actual_reward_list, label="actual_reward")
    # plt.plot(max_reward_list, label="max_reward")
    # plt.legend()
    # plt.show()

    end_time = time.time()
    print("Time taken: ", end_time - start_time)





