from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import umap
import pandas as pd

class Clusterer:
    def __init__(self, _list, method='kmeans',**kwargs):
        self.method_name = method.lower()
        self.model = None
        self.X = None
        self.params = kwargs
        self._list = _list
        self._init_model()

    def _init_model(self):
        if self.method_name == 'kmeans':
            self.model = KMeans(**self.params)
        elif self.method_name == 'dbscan':
            self.model = DBSCAN(**self.params)
        elif self.method_name == 'spectral':
            self.model = SpectralClustering(**self.params)
        else:
            raise ValueError(f"Unsupported method: {self.method_name}")

        
    def extract_features(self):
        for i, sig in enumerate(self._list):
            means = sig.mean(axis=1)         
            stds = sig.std(axis=1)
            features = np.concatenate([means, stds], axis=0)
            self._list[i] = np.expand_dims(features, axis=0)
        self.X =  np.concatenate(self._list, axis=0)

        
    def fit_predict(self):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        return self.model.fit_predict(X_scaled)

    def fit(self, X):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(self.X)
        if hasattr(self.model, "predict"):
            return self.model.predict(X_scaled)
        else:
            raise NotImplementedError("This clustering algorithm does not support predict()")

    def get_model(self):
        return self.model


class ClusterVisualizer:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
            )

    def plot(self, X, labels, title="Clustering Visualization"):
        X_2d = self.reducer.fit_transform(X)
        print (X_2d.shape)
        df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            #'cluster': labels.astype(str)
            'cluster': ['Vrai prénom' if labels[i] == 0 else 'Faux prénom' for i in range(len(labels))]
        })

        fig = px.scatter(
            df, x='x', y='y',
            color='cluster',
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=title,
            opacity=0.7,
            width=800,
            height=600
        )

        fig.update_layout(
            legend_title_text='Cluster',
            template='plotly_white'
        )
        fig.show()
