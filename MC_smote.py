from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import imbalanced_databases as imbd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


class MC_smote():
    def __init__(self, input, labels, K, k):
        """

        :param n: Number of minority class samples
        :param m: number of majority class samples
        :param K: K-means cluster number
        :param k: number of nearest neighbors
        """

        self.Xorg = input
        self.label_org = labels

        uniq, counts = np.unique(labels, return_counts=True)
        minority_idx = 1 if counts[0] > counts[1] else 0
        majority_idx = 0 if minority_idx == 1 else 1

        self.minority_label = uniq[minority_idx]
        self.majority_label = uniq[majority_idx]

        self.minority_num = counts[minority_idx]
        self.majority_num = counts[majority_idx]

        self.majority = input[labels == uniq[majority_idx]]
        self.minority = input[labels == uniq[minority_idx]]

        self.minority_labels = labels[labels == uniq[minority_idx]]
        self.majority_labels = labels[labels == uniq[majority_idx]]

        self.cluster_num = K
        self.neighbors = k

        n0 = self.majority_num - self.minority_num
        self.a = n0 // self.minority_num
        self.b = n0 % self.minority_num

    def cluster(self):
        # k-mean 聚类
        estimator = KMeans(n_clusters=self.cluster_num).fit(self.minority)

        centers = estimator.cluster_centers_
        l = np.unique(estimator.labels_)
        data = []
        y_label = []
        for _, label in enumerate(l):
            data.append(self.minority[estimator.labels_ == label])
            y_label.append(label)

        dic = dict(zip(y_label, data))  # print(dic[1])
        return centers, dic, estimator.labels_

    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        keth = np.random.rand(1)
        return x + (y - x) * keth

    def MC_generate(self):
        centers, dic, labels = self.cluster()

        # k-nn
        # fitting the model
        n_neighbors = min([len(centers), self.neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1)
        nn.fit(centers)
        _, ind = nn.kneighbors(centers)  # ind 为对应点周围数据的编号

        samples = []
        if self.a != 0:
            for i in range(self.a):
                for j in range(self.minority_num):
                    X = self.minority[j]
                    neighbors_clusters = ind[labels[j], 1:]

                    ramdomSelectKneighbor = np.random.permutation(neighbors_clusters)
                    cluster = dic[ramdomSelectKneighbor[0]]
                    m = cluster.shape[0]
                    permutation = list(np.random.permutation(m))
                    X_jk = cluster[permutation][0]
                    samples.append(self.sample_between_points(X, X_jk))

        ramdom_num_1ToN = np.random.permutation(range(self.minority_num))

        if self.b != 0:
            for i in range(self.b):
                k = ramdom_num_1ToN[i]
                X = self.minority[k]
                neighbors_clusters = ind[labels[k], 1:]

                ramdomSelectKneighbor = np.random.permutation(neighbors_clusters)
                cluster = dic[ramdomSelectKneighbor[0]]
                m = cluster.shape[0]
                permutation = list(np.random.permutation(m))
                X_jk = cluster[permutation][0,]
                samples.append(self.sample_between_points(X, X_jk))

        y = np.repeat(self.minority_label, len(samples))
        X_s = np.vstack([np.vstack(samples), self.Xorg])
        Y_s = np.hstack([y, self.label_org])
        return X_s, Y_s

    def MC_generate_num(self, num):

        centers, dic, labels = self.cluster()
        # k-nn
        # fitting the model
        n_neighbors = min([len(centers), self.neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1)
        nn.fit(centers)
        _, ind = nn.kneighbors(centers)  # ind 为对应点周围数据的编号 6个
        samples = []
        a = int(num / self.minority_num)
        b = np.mod(num, self.minority_num)

        if num != 0:
            for i in range(a):
                for j in range(self.minority_num):
                    X = self.minority[j,]
                    neighbors_clusters = ind[labels[j], 1:]

                    ramdomSelectKneighbor = np.random.permutation(neighbors_clusters)
                    cluster = dic[ramdomSelectKneighbor[0]]
                    m = cluster.shape[0]
                    permutation = list(np.random.permutation(m))
                    X_jk = cluster[permutation][0,]
                    samples.append(self.sample_between_points(X, X_jk))

        ramdom_num_1ToN = np.random.permutation(range(self.minority_num))
        if b != 0:
            for i in range(b):
                k = ramdom_num_1ToN[i]
                X = self.minority[k,]
                neighbors_clusters = ind[labels[k], 1:]

                ramdomSelectKneighbor = np.random.permutation(neighbors_clusters)
                cluster = dic[ramdomSelectKneighbor[0]]
                m = cluster.shape[0]
                permutation = list(np.random.permutation(m))
                X_jk = cluster[permutation][0,]
                samples.append(self.sample_between_points(X, X_jk))

        return np.array(samples)


if __name__ == '__main__':
    dataset = imbd.load_glass1()
    X, y1 = dataset['data'], dataset['target']
    # X = np.load('WindTurbine_X.npy')
    # y1 = np.load('WindTurbine_Y.npy')
    # 数据标准化
    X_Zscore = (X - np.mean(X, axis=0)) / np.sqrt(np.var(X, axis=0))
    S = MC_smote(input=X, labels=y1, K=12, k=5)
    dd, y = S.MC_generate()


    # plt.scatter(e[:, 1], e[:, 2], label='majority class', c='olive', marker='o')
    plt.scatter(dd[y == 1][:, 0], dd[y == 1][:, 1], label='majority class', c='red', marker='*')
    plt.scatter(X[y1 == 1][:, 0], X[y1 == 1][:, 1], label='minor class', c='blue', marker='.')

    plt.show()