from common.ldbc.utils import graph_from_cypher, saveGraph, loadGraph, visualizeGraph
import networkx as nx
import pandas as pd
import itertools
from collections import Counter
from matplotlib import pyplot as plt


def analyzeGraph(g):
    
    return {
        "#Nodes": len(g.nodes),
        "#Edges": len(g.edges),
        "AvgDegree": sum([g.degree[i] for i in g.nodes])/len(g.nodes),
        "MaxDegree": max(g.degree, key = lambda k: k[1])[1],
        "Diameter": nx.algorithms.distance_measures.diameter(g),
        "AvgClusteringCoefficient": nx.algorithms.cluster.average_clustering(g),
        "Nodes": g.nodes,
        "Edges": g.edges}

def count_nodes(all_nodes):
    unpacked_nodes = []
    for nodes in all_nodes:
        for node in nodes:
            unpacked_nodes.append(node)
    nodes_counted = Counter(unpacked_nodes)
    return nodes_counted

def intersection_analysis(df_train, df_test):
    print("## Intersection analysis")
    nodes_in_train_counted = count_nodes(df_train["Nodes"].values)
    nodes_in_test_counted = count_nodes(df_test["Nodes"].values)

    intersection = list(set(nodes_in_train_counted).intersection(set(nodes_in_test_counted)))
    
    print(f"There are {len(nodes_in_train_counted)} different nodes in the train datset and {len(nodes_in_test_counted)} nodes in the test dataset. {len(intersection)} nodes are includeded in both datasets.")
    
    sort_key = lambda node: nodes_in_train_counted[node] + nodes_in_test_counted[node]
    df_as_dict = []

    for node in sorted(intersection, reverse=True, key=sort_key):
        df_as_dict.append({"#train":nodes_in_train_counted[node], "#test": nodes_in_test_counted[node]})

    df = pd.DataFrame(df_as_dict)
    print(df.head().to_markdown())
    df.hist()
    plt.savefig("histogram.jpg")


if __name__ == "__main__":
    # Visualize sample
    graphs_train = [loadGraph("trainFeatures", str(i)+".pkl") for i in range(866)]
    graphs_test  = [loadGraph("testFeatures",  str(i)+".pkl") for i in range(866, 1081)]
    df_train = pd.DataFrame([analyzeGraph(g) for g in graphs_train])
    df_test = pd.DataFrame([analyzeGraph(g) for g in graphs_test])
    
    print("# Graph analysis")
    print("## Train")
    print(df_train.describe().to_markdown())
    print("## Test")
    print(df_test.describe().to_markdown())
    print("## Delta\nThis table shows weather the properties of training and test set are similar")
    print((df_train.describe()-df_test.describe()).to_markdown())
    intersection_analysis(df_train, df_test)
