import numpy

class FuzzyCMeans(object):

    def __init__(self, data_matrix, c, fuzzifier = 2, iterations = 300, epsilon = 10^(-3)):
        self.__c = c
        self.__fuzzifier = fuzzifier
        self.__data = numpy.array(data_matrix)
        self.__m = data_matrix.shape[0]
        self.__n = data_matrix.shape[1]
        self.__minimums = numpy.amin(data_matrix, axis = 0)
        self.__maximums = numpy.amax(data_matrix, axis = 0)
        self.__iterations = iterations
        self.__epsilon = epsilon
        self.weight_matrix = numpy.zeros((self.__m, self.__c))
        self.centroids = {i:numpy.zeros(self.__n) for i in range(c)}
        self.clusters = [0]*self.__m

    def __str__(self):
        line_1 = "Number of clusters: {0} \n".format(self.__c)
        assignment_string = ""
        for i in range(len(self.clusters)):
            if ((i+1)%3 != 0):
                assignment_string += "Sample {0}: Cluster {1}, ".format(i, self.clusters[i])
            else:
                assignment_string += "Sample {0}: Cluster {1}\n".format(i, self.clusters[i])
        line_2 = "Cluster assignment: \n"
        line_3 = assignment_string + "\n"
        line_4 = "Centroids: \n"
        centroid_string = ""
        for key in self.centroids.keys():
            list_string = [str(each) for each in self.centroids[key]]
            centroid = "<" + ", ".join(list_string) + "> "
            centroid_string += "Centroid {0}: {1} \n".format(key, centroid)
        return(line_1 + line_2 + line_3 + line_4 + centroid_string)

    def set_centroid(self, cluster, new_centroid):
        if (len(new_centroid) != len(self.centroids[cluster])):
            raise ValueError("Dimensions of centroids do not align. \n New centroid: {0} \n Old centroid: {1}".format(len(new_centroid), len(self.centroids[cluster])))
        if any(new_centroid < (self.__minimums - 1)) or any(new_centroid > (self.__maximums + 1)):
            raise ValueError("New centroid values outside of feature range.")
        self.centroids[cluster] = new_centroid

    def get_centroid(self, cluster):
        return(self.centroids[cluster])

    def centroids(self):
        return(self.centroids)

    def clusters(self):
        return(self.clusters)

    def weight_matrix(self):
        return(self.weight_matrix)

    def assign_clusters(self):
        self.clusters = [numpy.argmax(self.weight_matrix[i,:]) for i in range(self.__m)]

    def fit_model(self, initialize = True):
        count = 0
        if (initialize):
            self.__initialize_centroids()
        else:
            if numpy.sum(self.centroids.values) == 0:
                raise ValueError("All centroids sum to zero. Please initialze centroids.")
        centroid_change = numpy.ones(self.__c)*100
        self.__calculate_weight_matrix()
        while (count < self.__iterations) and all(centroid_change > self.__epsilon):
            centroid_change = self.__update_centroids()
            self.__calculate_weight_matrix()
            count += 1
        self.assign_clusters()

    def __initialize_centroids(self):
        random_matrix = numpy.random.rand(self.__c, self.__n)
        transform_matrix = numpy.diag(self.__maximums - self.__minimums)
        min_matrix = numpy.mat(numpy.ones((self.__c, self.__n)))*numpy.diag(self.__minimums)
        random_centroids = numpy.mat(random_matrix)*transform_matrix + min_matrix
        for i in range(self.__c):
            self.set_centroid(i, numpy.array(random_centroids)[i, :])

    def __calculate_weight_matrix(self):
        for i in range(self.__m):
            fuzz = 2/(self.__fuzzifier - 1)
            distance_array = numpy.array([numpy.linalg.norm(self.__data[i,:] - self.centroids[key])
              for key in self.centroids])**fuzz
            total_distance = sum(1/distance_array)
            for j in range(self.__c):
                cluster_distance = distance_array[j]
                weight = 1 / (cluster_distance * total_distance)
                if (weight > 1 or weight < 0):
                    raise ValueError("Weight matrix entry invalid: {0}".format(weight))
                else:
                    self.weight_matrix[i,j] = weight
            sample_sum = round(numpy.sum(self.weight_matrix[i, :]), 2)
            if (sample_sum > 1):
                raise ValueError("Total probability of cluster membership exceeds 1: {0}".format(sample_sum))

    def __update_centroids(self):
        old_centroids = self.centroids
        centroid_change = numpy.zeros(self.__c)
        fuzz = 2/(self.__fuzzifier - 1)
        for i in range(self.__c):
            cluster_weight = numpy.diag(self.weight_matrix[:,i])**fuzz
            numerator = numpy.sum(cluster_weight*numpy.mat(self.__data), axis = 0)
            # convert numerator from [1, self.__n] to [self.__n]
            numerator = numpy.array(numerator).reshape((self.__n))
            denomenator = numpy.sum(cluster_weight)
            self.set_centroid(i, numpy.array(numerator/denomenator))
            centroid_change[i] = numpy.dot(old_centroids[i] - self.centroids[i],
                                           old_centroids[i] - self.centroids[i])
        return(centroid_change)
