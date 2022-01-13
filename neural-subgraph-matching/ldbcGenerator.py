from neo4j import GraphDatabase
from common.ldbc.utils import graph_from_cypher, saveGraph, loadGraph
import networkx as nx

class LdbcGeneratorWithoutFeatures:

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
        result = tx.run("""MATCH (p:person)-[r:KNOWS*1..2]->(p2:person)
        WITH p, COUNT(DISTINCT p2) AS cnt
        WHERE cnt >= 20 AND cnt <=100
        RETURN p.person_id""")
        values = []
        for record in result:
            values.append(record.value())
        return values


    @staticmethod
    def innerQuery(tx,person_ids):
        resultContainer = []
        for id in person_ids:
            result = tx.run("MATCH (p1:person)-[r:KNOWS*1..2]->(p2:person) WHERE p1.person_id = $id RETURN *",id = id)
            resultContainer.append(graph_from_cypher(result))
        return resultContainer
## TODO:  Question: Is it necessary to separarate test and train set in terms of no overlapping person_ids

if __name__ == "__main__":
    generator = LdbcGeneratorWithoutFeatures("bolt://localhost:7687", "neo4j", "1234")
    targetGraphs = generator.get_target_graph()
    generator.close()

    train_set= len(targetGraphs)  * 0.8
    for graphId, targetGraph in enumerate(targetGraphs):
        if graphId < train_set:
            saveGraph("train",targetGraph,str(graphId))
        else:
            saveGraph("test",targetGraph,str(graphId))




#MATCH (p:person)-[r:KNOWS*1..2]->(p2:person)
#WITH p, COUNT(DISTINCT r) AS edgecount, COUNT(DISTINCT p2) AS cnt
#WHERE cnt >= 20 AND cnt <=100
#RETURN p.person_id, edgecount, cnt
#ORDER BY edgecount DESC, cnt DESC
