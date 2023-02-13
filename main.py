from rescan import HDBScan, DendrogramWithStability, SimplifiedDendrogram
import numpy as np
import sklearn.datasets as data
from sklearn.metrics.cluster import adjusted_rand_score
import hdbscan
from matplotlib import pyplot as plt
from random import randint, randrange

# A demo testing our own implementation of HDBSCAN against the reference implementation

def generate_random_data(seed = 42):
    moons, _ = data.make_moons(n_samples=50, noise=0.05, random_state=seed)
    ncenters = randint(2, 10)
    blobs, _ = data.make_blobs(
        n_samples=ncenters * 10 + randint(100, 200),
        centers=randint(2, 10),
        cluster_std=randrange(1, 10) / 10,
        random_state=seed,
    )
    test_data = np.vstack([blobs, moons])
    return test_data

def comparison_cluster_method(points):
    N = len(points)
    myscanner = HDBScan(points)
    t = DendrogramWithStability.starshaped(N)
    for u, v, d in list(myscanner.join_sequence()):
        t.try_join(u, v, d)
    t.condense()
    labels = t.extract_clusters()
    return labels

def comparison_cluster_method_v2(points):
    N = len(points)
    myscanner = HDBScan(points)
    t = SimplifiedDendrogram(N)
    for u, v, d in list(myscanner.join_sequence()):
        t.try_join(u, v, d)
    t.condense()
    labels = t.extract_clusters()
    return labels

def reference_cluster_method(points):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(points)
    return clusterer.labels_

def rand_index(labels_ref, labels_est):
    return adjusted_rand_score(labels_ref, labels_est)

if __name__ == "__main__":
    for _ in range(1):
        # Generating random data
        test_data = generate_random_data()
        print(f"N = {len(test_data)}")
        # Using the reference cluster method to get labels 
        labels_ref = reference_cluster_method(test_data)
        # Using comparison cluster method to get labels
        labels_est = comparison_cluster_method_v2(test_data)
        # Computing rand index using labels from reference and estimated clusters
        randindex = rand_index(labels_ref, labels_est)
        # Checking if rand index is less than 1, meaning the clusters are not the same
        if randindex < 1:
            # Creating dictionary to store RGB values for 20 possible labels 
            c_dict = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'black', 5: 'cyan', 6: 'magenta', 7: 'orange', 8: 'purple', 9: 'brown', 10: 'pink', 11: 'gray', 12: 'olive', 13: 'cyan', 14: 'magenta', 15: 'orange', 16: 'purple', 17: 'brown', 18: 'pink', 19: 'gray', 20: 'olive'}
            # Printing out the rand index and labels
            print("randindex: ", randindex)
            print("labels_ref: ", labels_ref)
            print("labels_est: ", labels_est)
            # # Generating a plot with two subplots to compare the reference and estimated clusters
            # fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            # ax[0].scatter(test_data.T[0], test_data.T[1], c=labels_est, alpha=0.5, s=80, linewidths=0)
            # ax[0].set_title("Estimated Clusters")
            # ax[1].scatter(test_data.T[0], test_data.T[1], c=labels_ref, alpha=0.5, s=80, linewidths=0)
            # ax[1].set_title("Reference Clusters")
            # plt.show()