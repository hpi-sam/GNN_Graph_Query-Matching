import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt

# Source from https://stackoverflow.com/questions/59289134/constructing-networkx-graph-from-neo4j-query-result

"""
edge features are not possible to use in the original implementation yet.
edge feature 1: KNOWS -> 0, IS_LOCATED_IN -> 1


node features
node feature 1: female person -> 0, male person -> 1, place -> 2

"""


def graph_from_cypher(results, useFeatures=False):
    G = nx.Graph()
    nodes = list(results.graph()._nodes.values())

    if not useFeatures:
        for node in nodes:
            G.add_node(node.id)
        rels = list(results.graph()._relationships.values())
        for rel in rels:
            G.add_edge(rel.start_node.id, rel.end_node.id)
        return G
    else:
        for node in nodes:
            G.add_node(node.id, x=node2feature(node))
            #nx.set_node_attributes(G, x = node2feature(node))
        rels = list(results.graph()._relationships.values())
        for rel in rels:
            G.add_edge(rel.start_node.id, rel.end_node.id)
        return G


def node2feature(node):
    if "person" in node.labels:
        gender = node.get("gender")
        if gender == "female":
            return 0
        if gender == "male":
            return 1
        raise Exception('unknown gender error.')
    if "place" in node.labels:
        return 2
    raise Exception(
        'unknown label error. node2feature method went wrong. please check.')


def saveGraph(setName, graph, name):
    directory = os.path.dirname('./data/'+setName+'/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(graph, open('./data/'+setName+'/' + name + '.pkl', 'wb'))


def loadGraph(setName, name):
    g = pickle.load(open('./data/'+setName+'/' + name, 'rb'))
    return g


def visualizeGraph(graph):
    labels = nx.get_node_attributes(graph, 'x')
    nx.draw(graph, labels=labels)
    plt.show()
