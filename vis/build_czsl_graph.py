from neo4j import GraphDatabase

class CZSLGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("数据库已清空。")

    def build_graph(self):
        with self.driver.session() as session:
            # 1. 创建所有节点 (Nodes)
            print("正在创建节点...")
            session.run("""
                // Seen Nodes
                CREATE (:SeenAttribute {name: 'ancient'})
                CREATE (:SeenObject {name: 'town'})
                CREATE (:SeenObject {name: 'clock'})
                CREATE (:SeenComposition {name: 'ancient town'})
                CREATE (:SeenComposition {name: 'ancient clock'})
                
                // Neighbor Nodes
                CREATE (:NeighAttribute {name: 'historic'})
                CREATE (:NeighAttribute {name: 'old'})
                CREATE (:NeighObject {name: 'city'})
                CREATE (:NeighComposition {name: 'historic city'})
                CREATE (:NeighComposition {name: 'old city'})
            """)

            # 2. 创建所有关系 (Edges) - 按照双向图构建
            print("正在创建拓扑关系...")
            session.run("""
                MATCH (a:SeenAttribute {name: 'ancient'})
                MATCH (o_town:SeenObject {name: 'town'})
                MATCH (o_clock:SeenObject {name: 'clock'})
                MATCH (c_town:SeenComposition {name: 'ancient town'})
                MATCH (c_clock:SeenComposition {name: 'ancient clock'})
                
                MATCH (na_hist:NeighAttribute {name: 'historic'})
                MATCH (na_old:NeighAttribute {name: 'old'})
                MATCH (no_city:NeighObject {name: 'city'})
                MATCH (nc_hist:NeighComposition {name: 'historic city'})
                MATCH (nc_old:NeighComposition {name: 'old city'})

                // R0, R1: Attribute