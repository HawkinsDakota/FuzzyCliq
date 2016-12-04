import numpy
from scipy.cluster import hierarchy
import FuzzyCMeans
import itertools

class HybridSnnCluster(object):

    def __init__(self, data, c = 2, fuzzifier = 2, iterations = 300, epsilon = 10**-3):
        if c < 2:
            raise ValueError("Lower bound of clusters must be higher than 2.")
        self.__c = c # user inputed lower bound for number of clusters
        self.K = None # number of clusters produced by snn cliq
        self.__current_clusters = 0 # number of current clusters
        self.__data = numpy.array(data) # array of data
        self.__m = data.shape[0] # number of samples
        self.__n = data.shape[1] # number of features
        self.__fuzzifier = fuzzifier # fuzz factor for c-means
        self.__iterations = iterations # max number of iterations for c-means
        self.__epsilon = epsilon # convergence error for c-means
        self.match_matrix = numpy.zeros((self.__m, self.__m)) # matrix noting the number of times each sample is clustered with another
        self.relationships = {i:[0] for i in range(self.__m)} # dictionary containing related samples -- denoted by cluster memberships.
        self.clusters = None # cluster assignment for each sample
        self.centroids = {} # dictionary containing centroids for each cluster
        self.cluster_history = None # (m x number of runs) matrix containing cluster assignment for each sample over every run

    def __snn_cliq(self):
        linkage = hierarchy.linkage(self.__data, method = 'ward')
        snn_clusters = hierarchy.cut_tree(linkage, 7)[:,0]
        self.__initialize_cluster_history(snn_clusters)
        self.__update_relationships()

    def __initialize_cluster_history(self, snn_clusters):
        self.K = len(set(snn_clusters))
        self.__current_clusters = self.K
        self.cluster_history = numpy.zeros((self.__m, self.K - self.__c + 1))
        self.cluster_history[:, 0] = snn_clusters
        self.clusters = numpy.array(snn_clusters)
        self.__calculate_centroids()

    def __update_relationships(self):
        for i in range(self.__current_clusters):
            clustered_samples = self.cluster_membership(i)
            for each in clustered_samples:
                non_identity = set(clustered_samples).difference(set([each]))
                self.match_matrix[each, clustered_samples] += 1
                self.relationships[each] = non_identity

    def __reset_centroids(self):
        self.centroids = {}

    def __calculate_centroids(self):
        for i in set(self.clusters):
            members = self.cluster_membership(i)
            cluster_subset = self.__data[members, :]
            self.__set_centroid(i ,numpy.mean(cluster_subset, axis = 0))
            
    def __set_centroid(self, cluster, centroid):
        if len(centroid) != self.__n:
            raise ValueError("Centroid dimensions do not match feature dimensions.")
        self.centroids[cluster] = centroid

    def __closest_centroids(self):
        if (len(self.centroids) == 0):
            raise ValueError("Centroids have not be initialized.")
        distance_matrix = numpy.zeros((self.__current_clusters, self.__current_clusters)) + numpy.diag([numpy.inf]*self.__current_clusters)
        comparisons = itertools.combinations(self.centroids, 2)
        for i,j in comparisons:
            norm = numpy.linalg.norm(self.centroids[i] - self.centroids[j])
            distance_matrix[i,j] = norm
            distance_matrix[j,i] = norm
        cluster1, cluster2 = numpy.where(distance_matrix == distance_matrix.min())[0]
        return cluster1, cluster2

    def __merge_closest_clusters(self):
        # find closest clusters by l2 norm from centroids
        cluster1, cluster2 = self.__closest_centroids()
        # find samples belonging to both clusters
        cluster1_members = set(self.cluster_membership(cluster1))
        cluster2_members = set(self.cluster_membership(cluster2))
        cluster_union = list(cluster1_members.union(cluster2_members))
        # re-assign samples to the same cluster      
        new_cluster = min(cluster1, cluster2)
        self.clusters[cluster_union] = new_cluster
        # decrement the current number of clusters
        self.__current_clusters -= 1
        # re-assign cluster names to prevent number skips
        unique_clusters = list(set(self.clusters))
        reassign_dict = {unique_clusters[i]:i for i in range(self.__current_clusters)}
        self.clusters = numpy.array([reassign_dict[each] for each in self.clusters])  
        # reset centroids to remove extra dictionary entry       
        self.__reset_centroids()
        # re-calculate new centroids
        self.__calculate_centroids()

    def cluster_membership(self, cluster):
        clustered_members = list(numpy.where(self.clusters == cluster)[0])
        return clustered_members

    def fit_model(self):
        print("Running SNNCliq to initialize clustering...")
        self.__snn_cliq()
        current_run = 1
        for i in range(self.K - 1, self.__c - 1, -1):
            print("Running Fuzzy C-means with c = {0}...".format(i))
            self.__merge_closest_clusters()
            fuzzy_clusters = FuzzyCMeans.FuzzyCMeans(self.__data, i, self.__fuzzifier,
              self.__iterations, self.__epsilon)
            for key in self.centroids:
                fuzzy_clusters.set_centroid(key, self.centroids[key])
            fuzzy_clusters.fit_model(initialize = False)
            self.clusters = numpy.array(fuzzy_clusters.clusters)
            self.centroids = fuzzy_clusters.centroids
            self.__update_relationships()
            print(self.__current_clusters)
            print(self.cluster_history.shape)
            print(current_run)
            self.cluster_history[:,current_run] = self.clusters
            current_run += 1
from pandas import read_csv
data = read_csv("~/GradSchool/2016-2017/Bradham/challenge2016/Data/quantileNormalized.txt", sep = "\t").T
b = HybridSnnCluster(data)
b.fit_model()