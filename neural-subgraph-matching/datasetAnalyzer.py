import sys
sys.path.append("/Users/nicolashoecker/Downloads/ldbcdataset/gnn_graph-counting_query-matching/neural-subgraph-matching/")
from common import data
from common import utils
import networkx as nx
from common import combined_syn
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from deepsnap.batch import Batch
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from common.ldbc.utils import visualizeGraph
import matplotlib.pyplot as plt
from common.ldbc.utils import loadGraph
from statistics import mean
import os
import random


dataset_size = 4096

def data2plots(dataset, caseSubfolderName):

    if not os.path.exists("./analysis/" + caseSubfolderName + "/"):
        os.makedirs("./analysis/" + caseSubfolderName + "/")

    histogram_deg = []
    histogram_numberNodes = []
    histogram_numberEdges = []
    histogram_average_clustering = []
    histogram_average_diameter = []
    histogram_average_radius = []
    for graph in dataset:
        degree_temp = []
        for v in graph:
            degree_temp.append(graph.degree[v])
        histogram_deg.append(mean(degree_temp))
        histogram_numberNodes.append(graph.number_of_nodes())
        histogram_numberEdges.append(graph.number_of_edges())
        histogram_average_clustering.append(nx.average_clustering(graph))
        histogram_average_diameter.append(nx.diameter(graph))
        histogram_average_radius.append(nx.radius(graph))


    plt.hist(histogram_deg, bins=25, alpha=0.5)
    plt.title('Node degree, avg:' + str(mean(histogram_deg)) )
    plt.xlabel('node degree')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName+ "/nodeDegreeHistogram" +'.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_numberNodes, bins=50, alpha=0.5)
    plt.title('#Nodes avg:' + str(mean(histogram_numberNodes)))
    plt.xlabel('#nodes')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName+  "/numberNodesHistogram" +'.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_numberEdges, bins=125, alpha=0.5)
    plt.title('#Edges avg:' + str(mean(histogram_numberEdges)))
    plt.xlabel('#edges')
    plt.ylabel('count')
    plt.savefig('./analysis/'  + caseSubfolderName+ "/numberEdgesHistogram" +'.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_average_clustering, bins=125, alpha=0.5)
    plt.title('#Average Clustering Coefficient avg:' + str(mean(histogram_average_clustering)))
    plt.xlabel('#avg clustering coefficient')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName+ "/avgClusteringCoeffHistogram" +'.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_average_diameter, bins=125, alpha=0.5)
    plt.title('Diameter avg:' + str(mean(histogram_average_diameter)))
    plt.xlabel('diameter')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName+ "/diameterHistogram" +'.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_average_radius, bins=125, alpha=0.5)
    plt.title('Radius avg:' + str(mean(histogram_average_radius)))
    plt.xlabel('radius')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName+ "/radiusHistogram" +'.png', bbox_inches='tight')
    plt.clf()


def case6():
    ## Increase edges only ##
    pass

def case5(increaseOfEdges):
    ## Increase edges only by percent in proportion to existing edges, increase from [0.0,1.0], if not enough non-edges exist, add maximum available ##
    data = combined_syn.get_dataset("graph", 4096, np.arange(5,30))
    dataloader = TorchDataLoader(data, collate_fn=Batch.collate([]), batch_size=4096, shuffle=False)

    modifiedGraphs = []
    for batch in dataloader:
        graphsInBatch = batch.G
        for graph in graphsInBatch:
            numberExistingEdges = graph.number_of_edges()
            numberEdgesToAdd = int(increaseOfEdges * numberExistingEdges)
            nonedges = list(nx.non_edges(graph))
            edgesToAdd = random.sample(nonedges, min(numberEdgesToAdd, len(nonedges)))
            for edge in edgesToAdd:
                graph.add_edge(edge[0], edge[1])
            modifiedGraphs.append(graph)

    data2plots(modifiedGraphs, "case5")


def case4():
    ## Decrease nodes only ##
    pass

def case3():
    ## Increase nodes only ##
    pass

def case2():
    ## Decrease edges and nodes ##
    pass

def case1():
    ## Increase edges and nodes ##
    data = combined_syn.get_dataset("graph", 4096, np.arange(5,60))
    dataloader = TorchDataLoader(data, collate_fn=Batch.collate([]), batch_size=4096, shuffle=False)
    data = [sample for batch in dataloader for sample in batch.G]
    data2plots(data, "case1")

def case0():
    ## Original ##
    data = combined_syn.get_dataset("graph", 4096, np.arange(5,30))
    dataloader = TorchDataLoader(data, collate_fn=Batch.collate([]), batch_size=4096, shuffle=False)
    data = [sample for batch in dataloader for sample in batch.G]
    data2plots(data, "case0")

if __name__ == "__main__":
    case0()
    case1()
    case5(0.2)
