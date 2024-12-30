import numpy as np

if __name__ == "__main__":
    # list all the json files in the logged_data folder, and load them into a dataframe
    import os
    import pandas as pd
    files = os.listdir("logged_data")
    files.sort()

    last_n = 10
    selected_logged_data_dfs = []
    for file in files[-last_n:]:
        selected_logged_data_dfs.append(pd.read_json(f"logged_data/{file}"))
    print(selected_logged_data_dfs)


    import matplotlib.pyplot as plt

    # plot all the agents in the selected_logged_data_df, in the same plot but with different colors

    
    plt.figure(figsize=(10, 5 * (last_n + 1)))
    
    # plot last n logged data on same plot vertically

    # agent name -> a list of mean watch duration over days across all runs
    total_watch_duration_list_per_agent = {} 
    max_expected_watch_duration_list = []
    for i, logged_data_df in enumerate(selected_logged_data_dfs):
        for agent in logged_data_df:
            if agent not in total_watch_duration_list_per_agent:
                total_watch_duration_list_per_agent[agent] = []
            total_watch_duration_list_per_agent[agent].append(sum(logged_data_df[agent]["actual_watch_duration_reward_list"][3:]))

    mean_watch_duration_list = {}
    std_watch_duration_list = {}
    confidence_interval_upper_bound_watch_duration_list = {}
    confidence_interval_lower_bound_watch_duration_list = {}

    for agent in total_watch_duration_list_per_agent:
        mean_watch_duration_list[agent] = np.mean(total_watch_duration_list_per_agent[agent])
        std_watch_duration_list[agent] = np.std(total_watch_duration_list_per_agent[agent])

    for agent_a in total_watch_duration_list_per_agent:
        for agent_b in total_watch_duration_list_per_agent:
            if agent_a != agent_b:
                # calculate the p value
                mean_a = mean_watch_duration_list[agent_a]
                mean_b = mean_watch_duration_list[agent_b]
                std_a = std_watch_duration_list[agent_a]
                std_b = std_watch_duration_list[agent_b]
                n_a = len(total_watch_duration_list_per_agent[agent_a])
                n_b = len(total_watch_duration_list_per_agent[agent_b])

                # calculate the t statistic
                t_statistic = (mean_a - mean_b) / np.sqrt(std_a**2 / n_a + std_b**2 / n_b)

                # calculate the p value
                from scipy.stats import t
                df = n_a + n_b - 2  # degrees of freedom
                p_value = 2 * (1 - t.cdf(abs(t_statistic), df))
                print(f"lift {(mean_a - mean_b) / mean_b} p value for {agent_a} and {agent_b}: {p_value}")