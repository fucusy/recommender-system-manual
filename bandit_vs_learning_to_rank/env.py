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
        
        # set up the duration for each title
        # sample a value from a normal distribution with mean 120 and std 10
        self.title_duration = np.random.normal(120, 10, self.num_titles)
        self.title_duration = np.clip(self.title_duration, 10, 1000)
        self.title_duration = np.round(self.title_duration)

        # set up two types of completion rate distribution parameters for each title
        # 1. the completion rate distribution parameters when the user has intention to finish the title
        # 2. the completion rate distribution parameters when the user has no intention to finish the title
        self.completion_rate_mean_with_intention = np.random.normal(0.9, 0.2, self.num_titles)
        self.completion_rate_mean_without_intention = np.random.normal(0.1, 0.01, self.num_titles)
        self.completion_rate_mean_with_intention = np.clip(self.completion_rate_mean_with_intention, 0, 1)
        self.completion_rate_mean_without_intention = np.clip(self.completion_rate_mean_without_intention, 0, 1)

        self.completion_rate_std_with_intention = np.random.normal(0.15, 0.01, self.num_titles)
        self.completion_rate_std_without_intention = np.random.normal(0.015, 0.001, self.num_titles)
        self.completion_rate_std_with_intention = np.clip(self.completion_rate_std_with_intention, 0, 1)
        self.completion_rate_std_without_intention = np.clip(self.completion_rate_std_without_intention, 0, 1)

        # set up finish intentation probability for each title
        self.finish_intention_probability = np.random.normal(0.5, 0.1, self.num_titles)
        self.finish_intention_probability = np.clip(self.finish_intention_probability, 0, 1)



    def get_title_list(self):
        return list(range(self.num_titles))

    def get_single_user_feedback(self, title_id):
        """Get click outcome for a shown title based on its probability.
        
        Args:
            title_id (int): ID of the title shown
            
        Returns:
            tuple: (click, watch_duration_minutes) : (bool, int), for example (True, 10)
        """
        if not 0 <= title_id < self.num_titles:
            raise ValueError(f"Title ID must be between 0 and {self.num_titles-1}")
        
        if np.random.random() < self.finish_intention_probability[title_id]:
            # the user has intention to finish the title
            completion_rate = np.random.normal(self.completion_rate_mean_with_intention[title_id], self.completion_rate_std_with_intention[title_id])
            completion_rate = np.clip(completion_rate, 0, 1)
            watch_duration_minutes = self.title_duration[title_id] * completion_rate
        else:
            # the user has no intention to finish the title
            completion_rate = np.random.normal(self.completion_rate_mean_without_intention[title_id], self.completion_rate_std_without_intention[title_id])
            completion_rate = np.clip(completion_rate, 0, 1)
            watch_duration_minutes = self.title_duration[title_id] * completion_rate
        
        clicked = np.random.random() < self.click_probabilities[title_id]

        return clicked, watch_duration_minutes


    def get_user_feedback(self, title_rankings):
        """Get user feedback for a list of title rankings.
        
        Args:
            title_rankings (list/array): List of title rankings
            
        Returns:
            array of the same length as title_rankings with CLICK, SKIP, or NOT_SEEN
        """
        feedback = []
        for title_id in title_rankings:
            clicked, watch_duration_minutes = self.get_single_user_feedback(title_id)
            if clicked:
                feedback.append(("CLICK", watch_duration_minutes))
            else:
                feedback.append(("SKIP", 0))
            if np.random.random() < self.exit_probability:
                break
        feedback.extend([("NOT_SEEN", 0)] * (len(title_rankings) - len(feedback)))
        return feedback
    
    def get_maximum_expected_reward(self):
        """
        The reward is a tuple with two elements, the first is the maximum expected clicks, the second is the maximum expected watch duration
        This function will return maximum reward at two different scenarios:
        1. the maximum expected clicks
        2. the maximum expected watch duration                
        """

        # calculate the expected watch duration for each title
        completion_rate_mean = self.finish_intention_probability * self.completion_rate_mean_with_intention + (1 - self.finish_intention_probability) * self.completion_rate_mean_without_intention
        expected_watch_duration = self.title_duration * completion_rate_mean * self.click_probabilities

        # scenarios 1: when the number of clicks is maximized
        # sort the click probabilities in descending order
        sorted_click_probabilities = sorted(enumerate(self.click_probabilities), key=lambda x: x[1], reverse=True)

        # loop through the sorted probabilities and calculate the expected reward
        expected_reward_clicks = 0
        expected_reward_watch_duration = 0
        seen_probability = 1
        for title_id, click_probability in sorted_click_probabilities:
            expected_reward_clicks += click_probability * seen_probability
            expected_reward_watch_duration += expected_watch_duration[title_id] * seen_probability
            seen_probability *= (1 - self.exit_probability)

        max_rewards_when_clicks_maximized = (expected_reward_clicks, expected_reward_watch_duration)


        # scenarios 2: when the watch duration is maximized
        # sort the expected watch durations in descending order
        sorted_watch_durations = sorted(enumerate(expected_watch_duration), key=lambda x: x[1], reverse=True)
        expected_reward_clicks = 0
        expected_reward_watch_duration = 0
        seen_probability = 1
        for title_id, watch_duration in sorted_watch_durations:
            expected_reward_clicks += self.click_probabilities[title_id] * seen_probability
            expected_reward_watch_duration += watch_duration * seen_probability
            seen_probability *= (1 - self.exit_probability)
        max_rewards_when_watch_duration_maximized = (expected_reward_clicks, expected_reward_watch_duration)
        return max_rewards_when_clicks_maximized, max_rewards_when_watch_duration_maximized

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
        """
        Collect user feedback for a given title ranking
        :param title_ranking: list of title ids that were shown to the user, e.g. [0, 1, 2, ...]
        :param user_feedback: list of tuple of feedback from user, action and watch duration, e.g. [("CLICK", 10), ("SKIP", 0), ("NOT_SEEN", 0), ...]
        """
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
        clean_user_feedback = [feedback for feedback, _ in user_feedback]
        self.update_theta(title_ranking, clean_user_feedback)

    def make_ranking(self):
        if np.random.rand() < self.epsilon:
            return np.random.permutation(self.num_titles)
        else:
            return self.make_exploitation_ranking()

    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("click_counts: ", self.click_counts[0:10])
        print("seen_counts: ", self.seen_counts[0:10])
        click_probabilities = self.click_counts / self.seen_counts
        print("click_probabilities: ", click_probabilities[0:10])

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
        clean_user_feedback = [feedback for feedback, _ in user_feedback]
        self.update_parameters(title_ranking, clean_user_feedback)

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
        print("click_counts: ", self.click_counts[0:10])
        print("click_through_rates: ", self.click_through_rates[0:10])

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
        self.valid_title_ids = [] # each element is a title id
        self.valid_user_feedback = [] # each element is a user feedback, 1 for CLICK, 0 for SKIP

    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback):
        clean_user_feedback = [feedback for feedback, _ in user_feedback]
        for title_id, feedback in zip(title_ids, clean_user_feedback):
            if feedback != "NOT_SEEN":
                self.valid_title_ids.append(title_id)
                if feedback == "CLICK":
                    self.valid_user_feedback.append(1)
                else:
                    self.valid_user_feedback.append(0)
            else:
                break
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.valid_title_ids) > 0:
            self.train()

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
        for title_id, feedbacks in zip(title_ids, user_feedback):
            feedback, watch_duration = feedbacks
            if feedback != "NOT_SEEN":
                clean_user_feedback.append(1 if feedback == "CLICK" else 0)
                clean_title_ids.append(title_id)
                if feedback == "CLICK":
                    click_count += 1
            else:
                break
        if len(clean_title_ids) > 1 and click_count > 0:
            self.clean_training_data.append((clean_title_ids, clean_user_feedback))
        
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.clean_training_data) > 0:
            self.train()


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
    title_size = 100
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

    reward_when_clicks_maximized, reward_when_watch_duration_maximized = env.get_maximum_expected_reward()
    print("reward_when_clicks_maximized: ", reward_when_clicks_maximized)
    print("reward_when_watch_duration_maximized: ", reward_when_watch_duration_maximized)
    max_reward_clicks = reward_when_clicks_maximized[0]
    max_reward_watch_duration = reward_when_watch_duration_maximized[1]

    for agent in agents:
        actual_click_reward_total = 0.0
        actual_watch_duration_reward_total = 0.0
        max_clicks_reward_total = 0.0
        max_watch_duration_reward_total = 0.0
        actual_clicks_reward_list = []
        actual_watch_duration_reward_list = []
        max_clicks_reward_list = []
        max_watch_duration_reward_list = []
        print("agent name: ", agent.__class__.__name__)
        for day in range(total_days):
            actual_clicks_the_day = 0.0
            actual_watch_duration_the_day = 0.0
            max_clicks_reward_the_day = 0.0
            max_watch_duration_reward_the_day = 0.0
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
                
                click_count = sum(1 for feedback, watch_duration in user_feedback if feedback == "CLICK")
                watch_duration = sum(watch_duration for feedback, watch_duration in user_feedback if feedback == "CLICK")

                actual_clicks_the_day += click_count
                actual_watch_duration_the_day += watch_duration

                max_clicks_reward_the_day += max_reward_clicks
                max_watch_duration_reward_the_day += max_reward_watch_duration

                agent.collect_user_feedback(title_ranking, user_feedback)
                
            actual_click_reward_total += actual_clicks_the_day
            max_clicks_reward_total += max_clicks_reward_the_day
            actual_watch_duration_reward_total += actual_watch_duration_the_day
            max_watch_duration_reward_total += max_watch_duration_reward_the_day
            actual_clicks_reward_list.append(actual_clicks_the_day)
            max_clicks_reward_list.append(max_clicks_reward_the_day)
            actual_watch_duration_reward_list.append(actual_watch_duration_the_day)
            max_watch_duration_reward_list.append(max_watch_duration_reward_the_day)
            
            print(f"Day {day}, actual_clicks_reward: {actual_clicks_the_day}, max_clicks_reward: {max_clicks_reward_the_day}, actual_watch_duration_reward: {actual_watch_duration_the_day}, max_watch_duration_reward: {max_watch_duration_reward_the_day}")
        agent.understand_agent()
        print("actual_clicks_reward (rounded): ", round(actual_click_reward_total, 2))
        print("max_clicks_reward (rounded): ", round(max_clicks_reward_total, 2))
        print("actual_watch_duration_reward (rounded): ", round(actual_watch_duration_reward_total, 2))
        print("max_watch_duration_reward (rounded): ", round(max_watch_duration_reward_total, 2))
    # import matplotlib.pyplot as plt
    # plt.plot(actual_reward_list, label="actual_reward")
    # plt.plot(max_reward_list, label="max_reward")
    # plt.legend()
    # plt.show()

    end_time = time.time()
    print("Time taken: ", end_time - start_time)





