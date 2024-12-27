## Setup
```bash
conda create -n recommender_env python=3.9
conda activate recommender_env
pip install -r requirements.txt
```

## Training

```bash
python bandit_vs_learning_to_rank/weighted_agents.py
```

Or 
```bash
python bandit_vs_learning_to_rank/agent.py
```

## Visualizing

```bash
python bandit_vs_learning_to_rank/visualize.py
```

![all_agents](../plots/all_agents.png)