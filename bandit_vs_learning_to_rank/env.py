import numpy as np

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
            # Generate random click probabilities using a normal distribution
            self.click_probabilities = np.clip(np.random.normal(0.1, 0.01, self.num_titles), 0, 1)

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



    def understand_env(self):
        """
        This function will print the click probabilities and the expected watch duration for each title
        """

        # collect the click probabilities into bins and print the number of titles in each bin
        bins = np.linspace(0, 0.2, 20)
        bin_indices = np.digitize(self.click_probabilities, bins)
        bin_counts = np.bincount(bin_indices)
        print(bin_counts)
        
        # do the same for the title duration
        bins = np.linspace(0, 1000, 100)
        bin_indices = np.digitize(self.title_duration, bins)
        bin_counts = np.bincount(bin_indices)
        print(bin_counts)



if __name__ == "__main__":
    title_size = 10000
    env = RecommenderEnv(num_titles=title_size)
    env.understand_env()

    # print the maximum expected reward
    max_rewards_when_clicks_maximized, max_rewards_when_watch_duration_maximized = env.get_maximum_expected_reward()
    print(f"Maximum expected reward when clicks are maximized: {max_rewards_when_clicks_maximized}")
    print(f"Maximum expected reward when watch duration is maximized: {max_rewards_when_watch_duration_maximized}")

    # print the manual edit ranking
    manual_edit_ranking = env.manual_edit_ranking()
    print(f"Manual edit ranking: {manual_edit_ranking}")

    # print the expected clicks and watch duration reward for the manual edit ranking
    expected_clicks_list = []
    expected_watch_duration_list = []
    for _ in range(1000):
        feedback = env.get_user_feedback(manual_edit_ranking)
        expected_clicks = np.sum([1 for action, _ in feedback if action == "CLICK"])    
        expected_watch_duration = np.sum([watch_duration for _, watch_duration in feedback])
        expected_clicks_list.append(expected_clicks)
        expected_watch_duration_list.append(expected_watch_duration)
    print(f"Expected clicks: {np.mean(expected_clicks_list)}")
    print(f"Expected watch duration: {np.mean(expected_watch_duration_list)}")

