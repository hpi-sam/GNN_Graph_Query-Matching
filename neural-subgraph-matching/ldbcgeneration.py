from neo4j import GraphDatabase
from utils import graph_from_cypher, saveGraph, loadGraph
import networkx as nx

class HelloWorldExample:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_target_graph(self):
        with self.driver.session() as session:
            uniquePersons = session.read_transaction(self.outerQuery)
            return session.read_transaction(self.innerQuery, uniquePersons)

    @staticmethod
    def outerQuery(tx):
        result = tx.run("MATCH (p:person) RETURN p.person_id LIMIT 2")
        values = []
        for record in result:
            values.append(record.value())
        return values


    @staticmethod
    def innerQuery(tx,person_ids):
        resultContainer = []
        for id in person_ids:
            result = tx.run("MATCH (p:person)-[r:KNOWS]->(p2:person) WHERE p.person_id = $id RETURN * LIMIT 3",id = id)
            resultContainer.append(graph_from_cypher(result))
            for record in result:
                print(record)
        return resultContainer
## TODO:  Question: Is it necessary to separarate test and train set in terms of no overlapping person_ids


if __name__ == "__main__":
    greeter = HelloWorldExample("bolt://localhost:7687", "neo4j", "1234")
    result = greeter.get_target_graph()
    greeter.close()
    #print(result)
    #print(type(result))
    sample_id = 0
    for subgraph in result:
        saveGraph("train",subgraph,str(sample_id))
        sample_id =+ 1
    #    print(subgraph)
    #    print(nx.density(subgraph))
    #    print(nx.to_dict_of_dicts(subgraph))

    testGraph = loadGraph("train", str(0))
    print(testGraph)
