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
    watch_duration_list_per_agent = {} 
    max_expected_watch_duration_list = []
    for i, logged_data_df in enumerate(selected_logged_data_dfs):
        for agent in logged_data_df:
            if agent not in watch_duration_list_per_agent:
                watch_duration_list_per_agent[agent] = []
            watch_duration_list_per_agent[agent].append(logged_data_df[agent]["actual_watch_duration_reward_list"])
            max_expected_watch_duration_list = logged_data_df[agent]["max_watch_duration_reward_list"]
    
    mean_watch_duration_list = {}
    
    for agent in watch_duration_list_per_agent:
        mean_watch_duration_list[agent] = np.mean(watch_duration_list_per_agent[agent], axis=0)

    agent_labels = {
        "TweedieModelAgent": "Tweedie Regression Model",
        "RegressionModelAgent": "Regression Model",
        "WeightedPointWiseModelAgent": "Weighted-Duration Pointwise Model",
        "PointWiseModelAgent": "Pointwise Model",
    }


    for i, logged_data_df in enumerate(selected_logged_data_dfs):
        colors = ['red', 'blue', 'green', 'orange', 'purple',  'brown', 'pink', 'gray']
        ax = plt.subplot(last_n+1, 1, i + 1)
        plotted_max_duration = False
        for agent in logged_data_df:                                   
            color = colors.pop(0)        
        # put clicks on the left y axis and watch duration on the right y axis        
        # plt.plot(selected_logged_data_df[agent]["actual_clicks_reward_list"], label=agent, color=color)
        # plt.plot(selected_logged_data_df[agent]["max_clicks_reward_list"], label=f"{agent} max", color=color)
            if not plotted_max_duration:
                ax.plot(logged_data_df[agent]["max_watch_duration_reward_list"], label=f"Expected max watch duration", color='black', linestyle='--')
                plotted_max_duration = True                
            ax.plot(logged_data_df[agent]["actual_watch_duration_reward_list"], label=f"{agent_labels[agent]}", color=color)
            # ax2.plot(logged_data_df[agent]["max_watch_duration_reward_list"], label=f"{agent} max watch duration", color=color)
        ax.set_ylabel("Total watch duration for 10,000 users")
        ax.set_xlabel("Day") # set the position of the x label to the bottom right
        
        ax.set_title(f"Run {i+1}")
        ax.legend(loc='upper left')
        ax.grid(True)
        # set margin
        ax.margins(x=0.01)

    ax = plt.subplot(last_n+1, 1, last_n+1)  
    # plot the mean watch duration and clicks for each agent
    colors = ['red', 'blue', 'green', 'orange', 'purple',  'brown', 'pink', 'gray']
    ax.plot(max_expected_watch_duration_list, label=f"Expected max watch duration", color='black', linestyle='--')
    for agent in mean_watch_duration_list:
        color = colors.pop(0)
        ax.plot(mean_watch_duration_list[agent], label=f"{agent_labels[agent]}", color=color)
        ax.set_ylabel("Total watch duration for all users")
        ax.set_title(f"Mean watch duration from all runs")
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.margins(x=0.01)

    # save the plot to a file
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/all_agents.pdf")
    # set the image quality to high
    plt.savefig(f"plots/all_agents.png", dpi=300) 
    plt.show()
    plt.close()
