# Importa as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt

# Define uma classe de clusterização K-means
class KMeanClustering:
    def __init__(self, k, max_iteracoes = 250):
        # O construtor: Inicializa o número de clusters (k) e os centróides
        self.k = k
        self.centroids = None
        self.max_iteracoes = max_iteracoes

    @staticmethod
    def distancia_euclidiana(data_points, centroids):
        # Método estático: Calcula a distância euclidiana entre pontos de dados e centróides
        return np.sqrt(np.sum((centroids - data_points)**2, axis=1))

    def fit(self, X):
        # O método de fit: Clusteriza os dados em X
        # Inicializa os centróides aleatoriamente dentro do intervalo dos dados
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(self.max_iteracoes):
            y = []

            # Atribui pontos de dados ao cluster mais próximo
            for data_point in X:
                distancias = KMeanClustering.distancia_euclidiana(data_point, self.centroids)
                cluster_num = np.argmin(distancias)
                y.append(cluster_num)

            y = np.array(y)

            cluster_indices = []

            # Encontra pontos de dados em cada cluster
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            # Calcula novos centros de cluster
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            # Verifica a convergência
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y

import seaborn as sns
import pandas as pd

def spectral_clustering(X, n_clusters, n_neighbors):
    
    A_full = np.sqrt(np.square(X[:, None, :] - X[None, :, :]).sum(axis=-1))

    A_sort = np.argsort(A_full, axis=0)[:n_neighbors]
    A = np.zeros_like(A_full)

    # Preenche as matrizes dos vizinhos mais proximos com um
    for i in range(A.shape[0]):
        A[i,A_sort[:,i]] = 1

    np.fill_diagonal(A, 0)
    
    # Matrix identidade
    I = np.zeros_like(A)
    np.fill_diagonal(I, 1)


    D = np.zeros_like(A)
    np.fill_diagonal(D, np.sum(A,axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)


    eigenvalues, eigenvectors = np.linalg.eig(L)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    
    # Orderana os eigenvectors de modo decrescente baseado na magnitude dos eigenvalues
    eigenvectors_sorted = eigenvectors[:,eigenvalues.argsort()]


    # Usando o KMeanClustering implementado.
    X_transformed = eigenvectors_sorted[:,0:2]
    kmeans = KMeanClustering(k=n_clusters)
    labels = kmeans.fit(X_transformed)
    
    plt.scatter(eigenvectors_sorted[:,0], eigenvectors_sorted[:,1])
    fig = plt.figure(figsize=(6, 6))

    df = pd.DataFrame(X, columns =['x', 'y']) 

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels)

    plt.savefig('X_label.png', bbox_inches='tight', dpi=600)

    return



# Gera dados sintéticos com 3 clusters
from sklearn import datasets

print("Gerando dados...")
n_samples = 100
seed = 30
X, _ = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)

print("Usando o KMean no dataset...")
# Cria uma instância de KMeanClustering com max iteracoes por padrao de 250
km = KMeanClustering(k=2)

# Aplica a clusterização K-means e obtém os rótulos dos clusters
labels = km.fit(X)

# Visualiza os pontos de dados e os centróides
fig = plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(km.centroids[:, 0], km.centroids[:, 1], c=range(len(km.centroids)), marker="*", s=200)
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
                labelbottom=False,labeltop=False,labelleft=False,labelright=False)
plt.savefig('kmean.png', bbox_inches='tight', dpi=600)


fig = plt.figure(figsize=(6, 6))
plt.scatter(X[:,0], X[:,1])
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
                labelbottom=False,labeltop=False,labelleft=False,labelright=False)
plt.savefig('data.png', bbox_inches='tight', dpi=600)

print("Aplicando o Spectral Cluster no dataset...")

spectral_clustering(X, n_clusters=2, n_neighbors=5)

print("Concluido com suceseso, verifique a pasta para obter os resultados.")
