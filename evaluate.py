import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def evaluate(embeddings, labels, test_size=0.2):
    """Evaluate embeddings on downstream tasks"""
    # Node classification
    indices = np.random.permutation(len(labels))
    split = int(len(indices) * (1 - test_size))
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings[indices[:split]], labels[indices[:split]])
    acc = clf.score(embeddings[indices[split:]], labels[indices[split:]])
    
    # Clustering
    sil_score = silhouette_score(embeddings, labels)
    
    # Visualization
    tsne = TSNE(n_components=2)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar()
    plt.title("t-SNE Visualization of Node Embeddings")
    plt.show()
    
    return {"accuracy": acc, "silhouette": sil_score}