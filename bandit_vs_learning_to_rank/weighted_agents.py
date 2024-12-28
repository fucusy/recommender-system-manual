import numpy as np
import torch
import torch.nn as nn
from loss import PairwiseHingeLoss, TweedieLoss
from env import RecommenderEnv
from agent import Agent, BanditAgent, BucketSortingAgent, PointWiseModelAgent, PairWiseModelAgent
from model import PointWiseModel, PairWiseModel

class BucketSortingWatchTimeAgent(Agent):
    """
    Bucket Sorting Agent
    """
    def __init__(self, num_titles):
        self.bucket_size = min(max(int(num_titles / 100), 1), 5)
        self.num_titles = num_titles
        self.watch_times = np.zeros(num_titles)
        self.seen_counts = np.zeros(num_titles)
        self.watch_time_per_seen = np.zeros(num_titles)
   
    def collect_user_feedback(self, title_ranking, user_feedback):
        self.update_parameters(title_ranking, user_feedback)

    def make_ranking(self):
        # sort by click count first, group them into 10 buckets, the clicks in each bucket are equal, means the number of clicks in each bucket is 10% of the total clicks
        sorted_watch_times = sorted(enumerate(self.watch_times), key=lambda x: x[1], reverse=True)
        buckets = [[] for _ in range(10)]
        total_watch_time = sum(self.watch_times)
        accumulated_watch_time = 0
        bucket_watch_time_quota = total_watch_time / self.bucket_size 
        for title_id, watch_time in sorted_watch_times:
            accumulated_watch_time += watch_time
            bucket_index = min(int(accumulated_watch_time / bucket_watch_time_quota), self.bucket_size - 1)
            buckets[bucket_index].append(title_id)
        
        # sort each bucket by click through rate
        for bucket in buckets:
            bucket.sort(key=lambda x: self.watch_time_per_seen[x], reverse=True)
        
        # flatten the buckets
        title_ranking = [title_id for bucket in buckets for title_id in bucket]
        return title_ranking

    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("bucket_size: ", self.bucket_size)
        print("watch_times: ", self.watch_times[0:10])
        print("watch_time_per_seen: ", self.watch_time_per_seen[0:10])

    def day_starts(self):
        pass

    def update_parameters(self, title_ids, user_feedback):
        """
        :param title_ids: list of title ids that were shown to the user, e.g. [0, 1, 2, ...]
        :param user_feedback: list of feedback tuple from user, e.g. [("CLICK", 10), ("SKIP", 0), ("NOT_SEEN", 0), ...]
        Update parameters based on user feedback
        """
        for title_id, feedback_tuple in zip(title_ids, user_feedback):
            feedback, watch_duration = feedback_tuple
            if feedback != "NOT_SEEN":
                if feedback == "CLICK":
                    self.watch_times[title_id] += watch_duration
                    self.seen_counts[title_id] += 1
                elif feedback == "SKIP":
                    self.seen_counts[title_id] += 1
                self.watch_time_per_seen[title_id] = self.watch_times[title_id] / self.seen_counts[title_id]
        

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


class WeightedPointWiseModelAgent(Agent):
    def __init__(self, num_titles):
        self.num_titles = num_titles        
        self.model = PointWiseModel(num_titles)
        self.valid_title_ids = [] # each element is a title id
        self.valid_user_feedback = [] # each element is a user feedback, 1 for CLICK, 0 for SKIP
        self.valid_watch_duration_weights = [] # each element is a watch duration
    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback): 
        watch_duration_sum = 0
        valid_size = 0
        for title_id, feedback_tuple in zip(title_ids, user_feedback):
            feedback, watch_duration = feedback_tuple
            watch_duration_sum += watch_duration            
            if feedback != "NOT_SEEN":
                valid_size += 1
                self.valid_title_ids.append(title_id)
                if feedback == "CLICK":
                    self.valid_user_feedback.append(1)
                else:
                    self.valid_user_feedback.append(0)                                
            else:
                break
        self.valid_watch_duration_weights.extend([watch_duration_sum / valid_size] * valid_size)
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.valid_title_ids) > 0:
            self.train()

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, valid_title_ids, valid_user_feedback, valid_watch_duration_weights):
                self.valid_title_ids = valid_title_ids
                self.valid_user_feedback = valid_user_feedback
                self.valid_watch_duration_weights = valid_watch_duration_weights

            def __len__(self):
                return len(self.valid_title_ids)

            def __getitem__(self, idx):
                title_ids = self.valid_title_ids[idx]
                user_feedback = self.valid_user_feedback[idx]
                watch_duration = self.valid_watch_duration_weights[idx]
                # Convert "SKIP" to 0 and "CLICK" to 1
                return torch.tensor(title_ids, dtype=torch.long), torch.tensor(user_feedback, dtype=torch.float32), torch.tensor(watch_duration, dtype=torch.float32)

        dataset = UserFeedbackDataset(self.valid_title_ids, self.valid_user_feedback, self.valid_watch_duration_weights)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(100):
            epoch_loss = 0.0
            data_size = 0
            for title_ids, user_feedback, watch_duration_weight in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_ids)
                loss = criterion(outputs, user_feedback) * watch_duration_weight
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_ids)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")



class WeightedPairWiseModelAgent(Agent):
    def __init__(self, num_titles):
        self.num_titles = num_titles        
        self.model = PairWiseModel(num_titles)
        self.clean_training_data = [] # each element is a tuple of (title_ids, list of 1 or 0)
        self.valid_watch_duration_weights = [] # each element is a watch duration

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
        watch_duration_sum = 0
        for title_id, feedbacks in zip(title_ids, user_feedback):
            feedback, watch_duration = feedbacks
            if feedback != "NOT_SEEN":
                clean_user_feedback.append(1 if feedback == "CLICK" else 0)
                clean_title_ids.append(title_id)
                if feedback == "CLICK":
                    click_count += 1
                watch_duration_sum += watch_duration
            else:
                break
        if len(clean_title_ids) > 1 and click_count > 0:
            self.clean_training_data.append((clean_title_ids, clean_user_feedback, watch_duration_sum))
        
    
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
                title_ids, user_feedback, watch_duration_sum = self.clean_training_data[idx]
                return torch.tensor(title_ids, dtype=torch.long), torch.tensor(user_feedback, dtype=torch.float32), torch.tensor(watch_duration_sum, dtype=torch.float32), torch.tensor(len(title_ids), dtype=torch.long)

        dataset = UserFeedbackDataset(self.clean_training_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for epoch in range(100):
            epoch_loss = 0.0
            data_size = 0
            for title_ids, user_feedback, watch_duration_sum, n in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_ids)[0].reshape(1, -1)
                user_feedback = user_feedback.reshape(1, -1)
                loss = criterion(outputs, user_feedback, n) / (n * n) * watch_duration_sum
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                data_size += len(title_ids)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, data points: {data_size}, Loss per data point: {epoch_loss / data_size}")

        print("Training completed")



class TweedieModelAgent(Agent):
    def __init__(self, num_titles):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.device = torch.device("cpu")
        print("Using device: ", self.device)
        self.num_titles = num_titles        
        self.model = PointWiseModel(num_titles).to(self.device)
        self.valid_title_ids_tensor = [] # each element is a title id
        self.valid_watch_duration_tensor = [] # each element is a watch duration

    def make_ranking(self):
        # Get scores for all titles
        scores = self.model(torch.arange(self.num_titles, device=self.device))
        # Return titles sorted by their scores (highest first)
        return torch.argsort(scores, descending=True)

    def collect_user_feedback(self, title_ids, user_feedback): 
        for title_id, feedback_tuple in zip(title_ids, user_feedback):
            feedback, watch_duration = feedback_tuple

            title_id_tensor = torch.tensor(title_id, dtype=torch.long, device=self.device)
            watch_duration_tensor = torch.tensor(watch_duration, dtype=torch.float32, device=self.device)
            if feedback != "NOT_SEEN":
                self.valid_title_ids_tensor.append(title_id_tensor)
                self.valid_watch_duration_tensor.append(watch_duration_tensor)
            else:
                break
    
    def understand_agent(self):
        print("agent name: ", self.__class__.__name__)
        print("model parameters: ", self.model.parameters())

    def day_starts(self):
        if len(self.valid_title_ids_tensor) > 0:
            self.train()

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # using Sigmoid Cross Entropy Loss first
        criterion = TweedieLoss(p=1.5)

        class UserFeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, valid_title_ids, valid_watch_duration, device):
                self.valid_title_ids = valid_title_ids
                self.valid_watch_duration = valid_watch_duration
                self.device = device

            def __len__(self):
                return len(self.valid_title_ids)

            def __getitem__(self, idx):
                title_ids = self.valid_title_ids[idx]
                watch_duration = self.valid_watch_duration[idx]
                # Convert "SKIP" to 0 and "CLICK" to 1
                return title_ids, watch_duration

        dataset = UserFeedbackDataset(self.valid_title_ids_tensor, self.valid_watch_duration_tensor, self.device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(100):
            epoch_loss = 0.0
            data_size = 0
            for title_ids, watch_duration in dataloader:
                optimizer.zero_grad()
                outputs = self.model(title_ids)
                loss = criterion(outputs, watch_duration)
                loss = loss.mean()
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
    watch_time_bucket_agent = BucketSortingWatchTimeAgent(num_titles=title_size)
    bucket_sorting_click_count_agent = BucketSortingAgent(num_titles=title_size)
    weighted_pointwise_agent = WeightedPointWiseModelAgent(num_titles=title_size)
    weighted_pairwise_agent = WeightedPairWiseModelAgent(num_titles=title_size)
    pairwise_agent = PairWiseModelAgent(num_titles=title_size)
    tweedie_agent = TweedieModelAgent(num_titles=title_size)

    # agents = [weighted_pairwise_agent, pairwise_agent, weighted_pointwise_agent, pointwise_agent, bandit, bucket_sorting_click_count_agent, watch_time_bucket_agent]
    agents = [tweedie_agent, weighted_pointwise_agent, pointwise_agent]
    agents = [tweedie_agent]

    # sort by probability, print the title id and value
    sorted_probabilities = sorted(enumerate(env.click_probabilities), key=lambda x: x[1], reverse=True)
    for title_id, probability in sorted_probabilities:
        print(f"title {title_id}: {round(probability, 2)}")

    reward_when_clicks_maximized, reward_when_watch_duration_maximized = env.get_maximum_expected_reward()
    print("reward_when_clicks_maximized: ", reward_when_clicks_maximized)
    print("reward_when_watch_duration_maximized: ", reward_when_watch_duration_maximized)
    max_reward_clicks = reward_when_clicks_maximized[0]
    max_reward_watch_duration = reward_when_watch_duration_maximized[1]

    logged_data = {}
    for agent in agents:
        logged_data[agent.__class__.__name__] = {"actual_clicks_reward_list": [], "actual_watch_duration_reward_list": [], "max_clicks_reward_list": [], "max_watch_duration_reward_list": []}

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

            logged_data[agent.__class__.__name__]["actual_clicks_reward_list"].append(actual_clicks_the_day)
            logged_data[agent.__class__.__name__]["max_clicks_reward_list"].append(max_clicks_reward_the_day)
            logged_data[agent.__class__.__name__]["actual_watch_duration_reward_list"].append(actual_watch_duration_the_day)
            logged_data[agent.__class__.__name__]["max_watch_duration_reward_list"].append(max_watch_duration_reward_the_day)   
            
            print(f"Day {day}, actual_clicks_reward: {actual_clicks_the_day}, max_clicks_reward: {max_clicks_reward_the_day}, actual_watch_duration_reward: {actual_watch_duration_the_day}, max_watch_duration_reward: {max_watch_duration_reward_the_day}")
        agent.understand_agent()
        print("actual_clicks_reward (rounded): ", round(actual_click_reward_total, 2))
        print("max_clicks_reward (rounded): ", round(max_clicks_reward_total, 2))
        print("actual_watch_duration_reward (rounded): ", round(actual_watch_duration_reward_total, 2))
        print("max_watch_duration_reward (rounded): ", round(max_watch_duration_reward_total, 2))

    # save the logged data to a json file, add timestamp to the filename 
    import pandas as pd
    df = pd.DataFrame(logged_data)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    
    # save it to a folder called logged_data, create the folder if it doesn't exist
    import os
    if not os.path.exists("logged_data"):
        os.makedirs("logged_data")
    import random
    random_seed = random.randint(0, 100000000)
    df.to_json(f"logged_data/logged_data_{timestamp}_{random_seed}.json", index=False)
    
    
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
