from neo4j import GraphDatabase
from common.ldbc.utils import graph_from_cypher, saveGraph, loadGraph
import networkx as nx

class LdbcGenerator:

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
        result = tx.run("MATCH (p:person) RETURN p.person_id LIMIT 1200")
        values = []
        for record in result:
            values.append(record.value())
        return values


    @staticmethod
    def innerQuery(tx,person_ids):
        resultContainer = []
        for id in person_ids:
            result = tx.run("MATCH (p:person)-[r:KNOWS*1..2]->(p2:person) WHERE p.person_id = $id RETURN * LIMIT 30",id = id)
            if (result.peek() != None):
                resultContainer.append(graph_from_cypher(result))
        return resultContainer
## TODO:  Question: Is it necessary to separarate test and train set in terms of no overlapping person_ids


def graphId2setName(graphId):
    if graphId > 1000:
        return "test"
    return "train"


if __name__ == "__main__":
    generator = LdbcGenerator("bolt://localhost:7687", "neo4j", "1234")
    targetGraphs = generator.get_target_graph()
    generator.close()

    for graphId, targetGraph in enumerate(targetGraphs):
        saveGraph(graphId2setName(graphId),targetGraph,str(graphId))
