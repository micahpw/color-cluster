#%% [Markdown]
### Title
# Color Clustering Image Demo
# Author: Micah Webb
#%%
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread("./data/IMG_0539.JPG")

subimage = im #[0:100][0:100] # Use smaller image if debugging

sub_shape = subimage.shape
print(sub_shape)
plt.imshow(im)

#%%

# Reshape to an array with 3 cols
# Then load into dataframe 
df = pd.DataFrame(subimage.reshape(-1,3))
df.columns = ['R', 'G', 'B']

#%% 

#Remove Correlations from the data.
from sklearn.decomposition import FactorAnalysis
X = df[['R','G','B']]

fac = FactorAnalysis(n_components=2,rotation='varimax')
out = fac.fit_transform(X)

#New DataFrame with 2 cols
df_out = pd.DataFrame(out)
df_out.columns = ['f1', 'f2']
df_out.plot(kind='hexbin',x='f1', y='f2', bins='log')
#%%

# Clustering Data
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, MiniBatchKMeans
import numpy as np

X_out = df_out[['f1','f2']].values
print('clustering')
#clust = AgglomerativeClustering(n_clusters=2)
clust = MiniBatchKMeans(n_clusters=3)
#clust = OPTICS(min_samples=50, min_cluster_size=0.05, cluster_method='dbscan', eps=0.5, n_jobs=-1)
clust.fit(X_out)

#clust = DBSCAN(eps=0.5, min_samples=10).fit(X_out)
print('finished')
print(clust.labels_)

df_out['Labels'] = clust.labels_
#%%
import seaborn as sns
cmap = sns.color_palette("hls",3)

sns.scatterplot(data=df_out.sample(n=500), x='f1', y='f2', hue='Labels',palette=cmap)
plt.show()
#%%
# Plot the Original vs Clustered Image.
out_image = df_out['Labels'].values.reshape(sub_shape[0],sub_shape[1],1)
fig, axs = plt.subplots(1,2)

# Plot the Old and new image
axs[0].imshow(im)
axs[1].imshow(out_image)

axs[0].set_title('Original Image')
axs[1].set_title('Clustered Image')

# Remove the Grid Lines
axs[0].xaxis.set_ticks([])
axs[1].xaxis.set_ticks([])
axs[0].yaxis.set_ticks([])
axs[1].yaxis.set_ticks([])

# Tighten the layout
plt.tight_layout()
plt.show()
# %%
