import os
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



if __name__ == '__main__':


    item_factors_path = 'models/item_factors.pkl'
    # item_factors_path = 'models/item_factors_nn.pkl'

    
    # Load item factors
    with open(item_factors_path, 'rb') as f:
        item_factors_raw = pickle.load(f)



    

    # Read raw text file
    item_file_path = './data/ml-100k/u.item'
    item_df = pd.read_csv(item_file_path, sep='|', header=None, encoding='latin-1', usecols=[0, 1], names=['item_id', 'title'])

    # Convert dataframe to dictionary
    item_id_to_title = item_df.set_index('item_id')['title'].to_dict()

    # Create mappings and np array for item factors
    item_ids = list(item_factors_raw.keys())
    item_idx_to_id = {i: item_id for i, item_id in enumerate(item_ids)}
    item_id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
    item_factors = np.array([item_factors_raw[item_id] for item_id in item_ids])

    # Perform t-SNE clustering
    tsne = TSNE(n_components=2, random_state=0)
    item_factors_2d = tsne.fit_transform(item_factors)

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(item_factors_2d[:, 0], item_factors_2d[:, 1], marker='o', s=5, edgecolor='k', linewidth=0.5)


    # Add item id notation
    for i, txt in enumerate(range(len(item_factors))):
        item_id = item_idx_to_id[i]
        item_title = item_id_to_title.get(item_id, item_id)
        txt = str(item_title) + " (" + str(item_id) + ")"
        plt.annotate(txt, (item_factors_2d[i, 0], item_factors_2d[i, 1]), fontsize=1, color='orange', alpha=0.5)  

    plt.title('t-SNE Clustering of Item Factors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # Get the filename from item_factors_path but with pdf ending
    pdf_filename = os.path.splitext(os.path.basename(item_factors_path))[0] + '.pdf'

    plt.savefig(pdf_filename)
    
    os.system('open -a Preview ' + pdf_filename)

