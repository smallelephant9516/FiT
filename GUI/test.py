import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

umap_2D = pd.DataFrame({
    'Dimension 1': np.random.rand(1,2000)[0],
    'Dimension 2': np.random.rand(1,2000)[0],
})

all_index = np.random.choice([0,1,2],size=2000)

cmap = plt.cm.tab20
colors = cmap(np.linspace(0, 1, len(set(all_index))))
plt.scatter(umap_2D['Dimension 1'],umap_2D['Dimension 2'],c=all_index,cmap=cmap,alpha=0.5,s=5)
handles, labels = [], []
for i, color in enumerate(colors):
    handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
    labels.append(f"cluster {i+1}")
plt.legend(handles, labels, title="Cluster index", loc='upper right')

plt.show()