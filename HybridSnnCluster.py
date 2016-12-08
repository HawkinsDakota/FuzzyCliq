"""
@author Dakota Hawkins
Last edited: December 7, 2016
"""

import numpy
import FuzzyCMeans
import itertools
import subprocess
from pandas import DataFrame
from sklearn import decomposition
import os

class HybridSnnCluster(object):

    def __init__(self, data, c = 2, fuzzifier = 2, iterations = 100, epsilon = 10**-3):
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
        self.centroid_history = {} # dictionary of dictionaries for centroids at each iteration.

    def __snn_cliq(self):
        #linkage = hierarchy.linkage(self.__data, method = 'ward')
        #snn_clusters = hierarchy.cut_tree(linkage, 7)[:,0]
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        tmp_data = "tmp/input_data.csv"
        numpy.savetxt(tmp_data, self.__data, fmt = '%1.7f', delimiter = ',')
        subprocess.call('./snncliq.sh')

        if not os.path.exists("tmp/clusters.txt"):
           raise IOError("Cluster file was not written.")
        self.clusters = numpy.loadtxt("tmp/clusters.txt", dtype = int)
        if len(self.clusters) != self.__m:
            raise ValueError("SNNCliq failed to return the appropriate number of cluster assignments. Consider re-running.")

        self.K = len(set(self.clusters))
        self.__current_clusters = self.K
        self.__reassign_clusters()
        self.__calculate_centroids()
        self.__initialize_cluster_history()
        os.remove("tmp/input_data.csv")
        os.remove("tmp/clusters.txt")
        os.remove("tmp/edge.txt")

    def __initialize_cluster_history(self):
        self.cluster_history = numpy.zeros((self.__m, self.K - self.__c + 1))
        self.cluster_history[:, 0] = self.clusters

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

    def __reassign_clusters(self):
        unique_clusters = list(set(self.clusters))
        reassign_dict = {unique_clusters[i]:i for i in range(self.__current_clusters)}
        self.clusters = numpy.array([reassign_dict[each] for each in self.clusters])


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
        self.__reassign_clusters()
        # reset centroids to remove extra dictionary entry
        self.__reset_centroids()
        # re-calculate new centroids
        self.__calculate_centroids()

    def cluster_membership(self, cluster):
        clustered_members = list(numpy.where(self.clusters == cluster)[0])
        return clustered_members

    def __run_c_means(self, C):
        c_means = FuzzyCMeans.FuzzyCMeans(self.__data, C, self.__fuzzifier,
          self.__iterations, self.__epsilon)
        for key in self.centroids:
            c_means.set_centroid(key, self.centroids[key])
        c_means.fit_model(initialize = False)
        self.clusters = numpy.array(c_means.clusters)
        self.centroids = c_means.centroids

    def fit_model(self):
        print("Running SNNCliq to initialize clustering...")
        self.__snn_cliq()
        self.__run_c_means(self.K)
        self.__initialize_cluster_history()
        self.centroid_history[0] = self.centroids
        current_run = 1
        for i in range(self.K - 1, self.__c - 1, -1):
            print("Running Fuzzy C-means with c = {0}...".format(i))
            self.__merge_closest_clusters()
            self.__run_c_means(i)
            self.__update_relationships()
            self.cluster_history[:,current_run] = self.clusters
            self.centroid_history[current_run] = self.centroids
            current_run += 1

    def calculate_metrics(self):
        probability_of_match = self.match_matrix / (self.K - self.__c + 1)
        cluster_probabilities = numpy.zeros((self.K, self.cluster_history.shape[1]))
        for run in range(self.cluster_history.shape[1]):
            current_grouping = self.cluster_history[:, run]
            cluster_averages = numpy.zeros(len(set(current_grouping)))
            i = 0
            for cluster in set(current_grouping):
                grouped = numpy.where(current_grouping == cluster)[0]
                average_probs = [numpy.mean(probability_of_match[each, grouped]) for each in grouped]
                cluster_averages[i] = numpy.mean(average_probs)
                i += 1
            #scaled_averages = cluster_averages / sum(cluster_averages)
            scaled_averages = cluster_averages #/ numpy.sum(numpy.triu(probability_of_match))
            cluster_probabilities[0:len(scaled_averages), run] = scaled_averages
        return(cluster_probabilities)

    def write_cluster_history(self, output_file, sep = ','):
        columns = ["Iter{0}".format(i + 1) for i in range(self.cluster_history.shape[1])]
        rows = ["Sample{0}".format(i + 1) for i in range(self.__m)]
        output = DataFrame(self.cluster_history, index = rows, columns = columns)
        output.to_csv(output_file, sep = sep)

    def members_at_iter(self, cluster, iter_step):
        return(numpy.where(self.cluster_history[:, iter_step] == cluster)[0])

    def get_centroid_history_pca_data(self):
        feature_space = decomposition.PCA()
        feature_space.fit(self.__data)
        combined_data = []
        for runs in self.centroid_history:
            current = self.centroid_history[runs]
            for centroid in current:
                centroid_pca = feature_space.transform(current[centroid].reshape(1,-1))
                current_row = numpy.hstack((numpy.array(runs).reshape((1,-1)),
                                            numpy.array(centroid).reshape((1,-1)),
                                            centroid_pca))
                samples = self.members_at_iter(centroid, runs)
                repeat_row = numpy.tile(current_row, len(samples)).reshape(len(samples), current_row.shape[1])
                if centroid == 0 and runs == 0:
                    combined_data = numpy.hstack((samples.reshape((len(samples), 1)), repeat_row))
                else:
                    new_rows =  numpy.hstack((samples.reshape((len(samples), 1)), repeat_row))
                    combined_data = numpy.vstack((combined_data, new_rows))

        first_columns = ['Sample', 'Iteration', 'Cluster']
        components = ["PC{0}".format(i+1) for i in range(combined_data.shape[1] - 3)]
        plot_data = DataFrame(combined_data,
                              columns = first_columns + components)
        for each in first_columns:
            plot_data[each] = plot_data[each].astype('int')
        return(plot_data)

    def centroid_history_as_dataframe(self):
        if len(self.centroid_history) == 0:
            raise ValueError("Centroid history not instanstiated. Must call 'fit_model()' first.")
        combined_data = []
        for runs in self.centroid_history:
            current = self.centroid_history[runs]
            for centroid in current:
                centroid_pca = current[centroid].reshape(1,-1)
                current_row = numpy.hstack((numpy.array(runs).reshape((1,-1)),
                                            numpy.array(centroid).reshape((1,-1)),
                                            centroid_pca))
                if centroid == 0 and runs == 0:
                    combined_data = current_row
                else:
                    combined_data = numpy.vstack((combined_data, current_row))

        first_columns = ['Iteration', 'Cluster']
        components = ["Feature{0}".format(i+1) for i in range(combined_data.shape[1] - 2)]
        centroid_df = DataFrame(combined_data,
                              columns = first_columns + components)
        for each in first_columns:
            centroid_df[each] = centroid_df[each].astype('int')
        return(centroid_df)



#test_data = numpy.ones((150, 5))
#test_data[0:25,:] *= numpy.random.normal(0,1, (25,5))
#test_data[25:50,:] *= numpy.random.normal(-4,1, (25,5))
#test_data[50:75,:] *= numpy.random.normal(2,1, (25,5))
#test_data[75:100,:] *= numpy.random.normal(4,1, (25,5))
#test_data[100:125,:] *= numpy.random.normal(1,1, (25,5))
#test_data[125:150,:] *= numpy.random.normal(-2,1, (25,5))
#c = HybridSnnCluster(test_data)
#c.fit_model()
#huh = c.centroid_history_as_dataframe()

#from pandas import read_csv
#data = read_csv("/home/dakota/Documents/School/2016-2017/challenge2016/Data/quantileNormalized.txt", sep = "\t").T
#b = HybridSnnCluster(data)
#b.fit_model()
