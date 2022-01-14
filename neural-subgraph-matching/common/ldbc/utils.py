import networkx as nx
import pickle
import os

# Source from https://stackoverflow.com/questions/59289134/constructing-networkx-graph-from-neo4j-query-result

"""
edge features
edge feature 1: KNOWS -> 0, IS_LOCATED_IN -> 1


node features
node feature 1: female person -> 0, male person -> 1, place -> 2

"""

def graph_from_cypher(results, useFeatures = False):
    G = nx.Graph()
    nodes = list(results.graph()._nodes.values())

    if !useFeatures:
        for node in nodes:
            G.add_node(node.id)
        rels = list(results.graph()._relationships.values())
        for rel in rels:
            G.add_edge(rel.start_node.id, rel.end_node.id)
        return G
    else:
        for node in nodes:
            G.add_node(node.id,  # , labels=node._labels, properties=node._properties
                       )
        rels = list(results.graph()._relationships.values())
        for rel in rels:
            G.add_edge(rel.start_node.id, rel.end_node.id
            #,key=rel.id, type=rel.type, properties=rel._properties
            )
            #G.add_edge(rel.end_node.id,rel.start_node.id
            #,key=rel.id, type=rel.type, properties=rel._properties
            #)
        return G

def node2feature(node):



def saveGraph(setName, graph, name):
    directory = os.path.dirname('./data/'+setName+'/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(graph, open('./data/'+setName+'/' + name + '.pkl', 'wb'))


def loadGraph(setName, name):
    g = pickle.load(open('./data/'+setName+'/' + name, 'rb'))

    # edges = []

    # for edge in g.edges:
    #     u, v, k = edge
    #     props = g.edges[u, v, k]
    #     print(props)

    # print(g.edges)
    return g
