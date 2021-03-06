from neo4j import GraphDatabase
from common.ldbc.utils import graph_from_cypher, saveGraph, loadGraph, visualizeGraph
import networkx as nx


class LdbcGenerator:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_target_graph(self, useFeatures=False):
        with self.driver.session(database="snd") as session:
            uniquePersons = session.read_transaction(self.outerQuery)
            return session.read_transaction(self.innerQuery, uniquePersons, useFeatures=useFeatures)

    @staticmethod
    def outerQuery(tx):
        result = tx.run("""MATCH (p:person)-[r:KNOWS*1..2]->(p2:person)
        WITH p, COUNT(DISTINCT p2) AS cnt
        WHERE cnt >= 20 AND cnt <=100
        RETURN p.person_id""")
        values = []
        for record in result:
            values.append(record.value())
        print(len(values))
        return values

    @staticmethod
    def innerQuery(tx, person_ids, useFeatures):
        resultContainer = []
        for id in person_ids:
            #result = tx.run("MATCH (p1:person)-[r:KNOWS*1..2]->(p2:person) WHERE p1.person_id = $id RETURN *",id = id)
            result = tx.run(
                "MATCH (place1:place)<-[l1:IS_LOCATED_IN]-(p1:person)-[r:KNOWS*1..2]->(p2:person)-[l2:IS_LOCATED_IN]->(place2:place) WHERE p1.person_id = $id RETURN *", id=id)
            resultContainer.append(graph_from_cypher(
                result, useFeatures=useFeatures))
        return resultContainer
# TODO:  Question: Is it necessary to separarate test and train set in terms of no overlapping person_ids


if __name__ == "__main__":

    generator = LdbcGenerator("bolt://localhost:11003", "neo4j", "1234")
    targetGraphs = generator.get_target_graph(useFeatures=False)
    # generator.close()
    train_set = len(targetGraphs) * 0.8
    for graphId, targetGraph in enumerate(targetGraphs):
        if graphId < train_set:
            saveGraph("train", targetGraph, str(graphId))
        else:
            saveGraph("test", targetGraph, str(graphId))

    targetGraphs = generator.get_target_graph(useFeatures=True)
    generator.close()
    train_set = len(targetGraphs) * 0.8
    for graphId, targetGraph in enumerate(targetGraphs):
        if graphId < train_set:
            saveGraph("trainFeatures", targetGraph, str(graphId))
        else:
            saveGraph("testFeatures", targetGraph, str(graphId))

    # Visualize sample
    # g = loadGraph("trainFeatures", "1.pkl")
    # visualizeGraph(g)
