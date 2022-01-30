import sys
# sys.path.append("/Users/nicolashoecker/Downloads/ldbcdataset/gnn_graph-counting_query-matching/neural-subgraph-matching/")
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
from os import listdir
from os.path import isfile, join
import random
import pandas as pd
from statistics import mean
from subgraph_matching.train import main as train_model
import wandb

dataset_size = 4096


def data2plots(dataset, caseSubfolderName):

    data = [sample for batch in dataset for sample in batch.G]

    if not os.path.exists("./analysis/" + caseSubfolderName + "/"):
        os.makedirs("./analysis/" + caseSubfolderName + "/")

    histogram_deg = []
    histogram_numberNodes = []
    histogram_numberEdges = []
    histogram_average_clustering = []
    histogram_average_diameter = []
    histogram_average_radius = []
    histogram_eccentricities = []

    for graph in data:
        degree_temp = []
        for v in graph:
            degree_temp.append(graph.degree[v])
        histogram_deg.append(mean(degree_temp))
        histogram_numberNodes.append(graph.number_of_nodes())
        histogram_numberEdges.append(graph.number_of_edges())
        histogram_average_clustering.append(nx.average_clustering(graph))
        histogram_average_diameter.append(nx.diameter(graph))
        histogram_average_radius.append(nx.radius(graph))
        histogram_eccentricities.append(list(nx.eccentricity(graph).values()))

    csv = pd.DataFrame(data={'node degree': histogram_deg, 'node number': histogram_numberNodes, 'edge number': histogram_numberEdges, 'eccentricities': histogram_eccentricities,
                             'average clustering coefficient': histogram_average_clustering, 'average diameter': histogram_average_diameter, 'average radius': histogram_average_radius})
    csv.to_csv('./analysis/' + caseSubfolderName + "/statistics.csv")

    plt.hist(histogram_deg, bins=25, alpha=0.5)
    plt.title('Node degree, avg:' + str(mean(histogram_deg)))
    plt.xlabel('node degree')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName +
                "/nodeDegreeHistogram" + '.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_numberNodes, bins=50, alpha=0.5)
    plt.title('#Nodes avg:' + str(mean(histogram_numberNodes)))
    plt.xlabel('#nodes')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName +
                "/numberNodesHistogram" + '.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_numberEdges, bins=125, alpha=0.5)
    plt.title('#Edges avg:' + str(mean(histogram_numberEdges)))
    plt.xlabel('#edges')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName +
                "/numberEdgesHistogram" + '.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_average_clustering, bins=125, alpha=0.5)
    plt.title('#Average Clustering Coefficient avg:' +
              str(mean(histogram_average_clustering)))
    plt.xlabel('#avg clustering coefficient')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName +
                "/avgClusteringCoeffHistogram" + '.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_average_diameter, bins=125, alpha=0.5)
    plt.title('Diameter avg:' + str(mean(histogram_average_diameter)))
    plt.xlabel('diameter')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName +
                "/diameterHistogram" + '.png', bbox_inches='tight')
    plt.clf()

    plt.hist(histogram_average_radius, bins=125, alpha=0.5)
    plt.title('Radius avg:' + str(mean(histogram_average_radius)))
    plt.xlabel('radius')
    plt.ylabel('count')
    plt.savefig('./analysis/' + caseSubfolderName +
                "/radiusHistogram" + '.png', bbox_inches='tight')
    plt.clf()

    eccentricity_histograms = [[histogram.count(eccentricity) for eccentricity in range(
        0, 30)] for histogram in histogram_eccentricities]
    averaged_eccentricity_histogram = [
        *map(mean, zip(*eccentricity_histograms))]
    average_of_averaged_eccentricity_histogram = mean(
        map(lambda x: float(x) * averaged_eccentricity_histogram[x], np.arange(0, 30)))

    plt.plot(np.arange(0, 30), averaged_eccentricity_histogram)
    plt.title('#Eccentricity avg:' +
              str(average_of_averaged_eccentricity_histogram))
    plt.xlabel('eccentricity')
    plt.xticks(np.arange(0, 30, step=1.0))
    plt.ylabel('average counted in graphs')
    plt.grid()
    plt.savefig('./analysis/' + caseSubfolderName +
                "/eccentricity.png", bbox_inches='tight')
    plt.clf()

    for i, graph in enumerate(random.sample(data, 5)):
        graph = data[0]
        labels = nx.get_node_attributes(graph, 'x')
        nx.draw(graph, labels=labels)
        plt.savefig('./analysis/' + caseSubfolderName +
                    "/testGraph" + str(i) + ".png", bbox_inches='tight')
        plt.clf()


def caseERGeneratorNodes(target_graph_size, shift_parameter, experiment_run_number):
    # target_graph_size defines the number of nodes for the original baseline target graph
    # shift_parameter defines the increase or decrease of number of nodes as a factor with respect to the original number of nodes
    # the number of edges are rescaled to stay the same

    # create dataset for experiment
    try:
        data = combined_syn.get_dataset("graph", 4096, np.arange(
            target_graph_size, target_graph_size+1), generatorName="ERNODES", shift_parameter=shift_parameter)
    except:
        print("invalid shift parameter")
        return None
    dataloader = TorchDataLoader(data, collate_fn=Batch.collate(
        []), batch_size=4096, shuffle=False)
    # save statistics
    experiment_name = "caseERGeneratorNodes" + "_targetGraphSize_"+str(target_graph_size) + "_shiftParameter_"+str(shift_parameter) + \
        "_experimentRun_" + str(experiment_run_number)
    data2plots(dataloader, experiment_name)
    # return dataloader
    return dataloader, experiment_name


def caseERGeneratorEdges(target_graph_size, shift_parameter, experiment_run_number):
    # target_graph_size defines the number of nodes
    # shift_parameter defines the increase or decrease of number of edges as a factor with respect to the original number of edges

    # create dataset for experiment
    try:
        data = combined_syn.get_dataset("graph", 4096, np.arange(
            target_graph_size, target_graph_size+1), generatorName="EREDGES", shift_parameter=shift_parameter)
    except:
        print("invalid shift parameter")
        return None
    dataloader = TorchDataLoader(data, collate_fn=Batch.collate(
        []), batch_size=4096, shuffle=False)
    # save statistics
    experiment_name = "caseERGeneratorEdges" + "_targetGraphSize_"+str(target_graph_size) + "_shiftParameter_"+str(shift_parameter) + \
        "_experimentRun_" + str(experiment_run_number)
    data2plots(dataloader, experiment_name)
    # return dataloader
    return dataloader, experiment_name


def caseWSGeneratorEccentricity(target_graph_size, shift_parameter, experiment_run_number):
    # target_graph_size defines the number of nodes
    # shift_parameter influences the eccentricity via increase/decrease of connected nodes

    # create dataset for experiment
    try:
        data = combined_syn.get_dataset("graph", 4096, np.arange(
            target_graph_size, target_graph_size+1), generatorName="WSECCENTRICITY", shift_parameter=shift_parameter)
    except:
        print("invalid shift parameter")
        return None
    dataloader = TorchDataLoader(data, collate_fn=Batch.collate(
        []), batch_size=4096, shuffle=False)
    # save statistics
    experiment_name = "caseWSGeneratorEccentricity" + "_targetGraphSize_"+str(target_graph_size) + "_shiftParameter_"+str(shift_parameter) + \
        "_experimentRun_" + str(experiment_run_number)
    data2plots(dataloader, experiment_name)
    # return dataloader
    return dataloader, experiment_name


def setup_remote_logging(args):
    wandb.init(project="neural-subgraph-matching-experiments",
               entity="neural-subgraph-matching", reinit=True)
    experiment = args['experiment'].split('_')
    wandb.config.update({
        'query_size': args['query_size'],
        'target_size': args['target_size'],
        'batches': args['batches'],
        'shift_param': args['shift_param'],
        'experiment_run': experiment[-1],
        'generator': experiment[0]
    })
    wandb.run.name = args['experiment']
    analysis = [f for f in listdir(
                './analysis/' + args['experiment'] + "/") if isfile(join('./analysis/' + args['experiment'] + "/", f))]
    for f in analysis:
        wandb.save('./analysis/' + args['experiment'] + "/" + f)


def train(loader, target_size, experiment, shift_param):
    query_size = 5
    batches_per_experiment = 6000
    args = {'query_size': query_size, 'target_size': target_size, 'dataset': loader.dataset,
            'experiment': experiment, 'shift_param': shift_param, 'batches': batches_per_experiment}
    setup_remote_logging(args)
    train_model(exp_args=args)


if __name__ == "__main__":
    target_graph_sizes = np.arange(5, 30, 1)
    shiftParameter_caseERGeneratorNodes = np.arange(1.0, 2.0, 0.2)
    shiftParameter_caseERGeneratorEdges = np.arange(1.0, 2.0, 0.2)
    shiftParameter_caseWSGeneratorEccentricity = np.arange(1.0, 2.0, 0.2)
    experimentRuns = np.arange(1, 2)

    for experiment in experimentRuns:
        for size in target_graph_sizes:
            for shift_param in shiftParameter_caseWSGeneratorEccentricity:
                print(
                    f"size: {size}, shiftParameter: {shift_param}, eccentricity")
                loader, experiment_name = caseWSGeneratorEccentricity(
                    size, shift_param, experiment)
                train(loader, size, experiment_name, shift_param)
                if 1./shift_param == 1.0:
                    continue
                print(
                    f"size: {size}, shiftParameter: {1.0/shift_param}, eccentricity")
                loader, experiment_name = caseWSGeneratorEccentricity(
                    size, 1.0/shift_param, experiment)
                train(loader, size, experiment_name, 1.0/shift_param)

            for shift_param in shiftParameter_caseERGeneratorNodes:
                print(f"size: {size}, shiftParameter: {shift_param}, nodes")
                loader, experiment_name = caseERGeneratorNodes(
                    size, shift_param, experiment)
                train(loader, size, experiment_name, shift_param)
                if 1./shift_param == 1.0:
                    continue
                print(
                    f"size: {size}, shiftParameter: {1.0/shift_param}, nodes")
                loader, experiment_name = caseERGeneratorNodes(
                    size, 1.0/shift_param, experiment)
                train(loader, size, experiment_name, 1.0/shift_param)

            for shift_param in shiftParameter_caseERGeneratorEdges:
                print(f"size: {size}, shiftParameter: {shift_param}, edges")
                loader, experiment_name = caseERGeneratorEdges(
                    size, shift_param, experiment)
                train(loader, size, experiment_name, shift_param)
                if 1./shift_param == 1.0:
                    continue
                print(
                    f"size: {size}, shiftParameter: {1.0/shift_param}, edges")
                loader, experiment_name = caseERGeneratorEdges(
                    size, 1.0/shift_param, experiment)
                train(loader, size, experiment_name, 1.0/shift_param)


# edge increase/decrease: increase/decrease the beta distribution by a factor p, so that mean is between [0,1]
# node increase/decrease: increase/decrease nodes, remove/add randomly edges until desired number of edges (structure preserving)
# eccentricity increase/decrease

# Experiment Parameter
# 1) shift_parameter
# 2) target graph size
# 3) number of experiment repetition
