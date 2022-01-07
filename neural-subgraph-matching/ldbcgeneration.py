from neo4j import GraphDatabase

class HelloWorldExample:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_target_graph(self):
        with self.driver.session() as session:
            uniquePersons = session.read_transaction(self.outerQuery)
            session.read_transaction(self.innerQuery, uniquePersons)

    @staticmethod
    def outerQuery(tx):
        result = tx.run("MATCH (p:person) RETURN p.person_id LIMIT 2")
        values = []
        for record in result:
            values.append(record.value())
        return values

    @staticmethod
    def innerQuery(tx,person_ids):
        for id in person_ids:
            result = tx.run("MATCH (p:person)-[r:KNOWS]->(p2:person) WHERE p.person_id = $id RETURN * LIMIT 3",id = id)
            for record in result:
                print(record)


if __name__ == "__main__":
    greeter = HelloWorldExample("bolt://localhost:7687", "neo4j", "1234")
    result = greeter.get_target_graph()
    greeter.close()
