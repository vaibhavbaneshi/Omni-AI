from utils.logger import logger
from utils.llm import llm
from configs.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from utils.text_cleaner import text_cleaner, clean_cypher
from pyvis.network import Network
from neo4j import GraphDatabase

import streamlit as st
import ast


class kg_service:

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            logger.info("✅ Connected to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Failed to establish a connection: {e}")
            self.driver = None

    # ----------------- Helpers -----------------
    def parse_triples(self, raw_triples: str):
        """Parse triples from LLM response (supports optional alias lists)."""
        try:
            triples = ast.literal_eval(raw_triples)
            if isinstance(triples, list):
                clean_triples = []
                alias_map = {}
                for item in triples:
                    if len(item) >= 3:
                        head, relation, tail = item[:3]
                        clean_triples.append((head, relation, tail))

                        # if 4th element exists, treat it as alias list
                        if len(item) == 4 and isinstance(item[3], list):
                            alias_map[head] = item[3]
                return clean_triples, alias_map
        except Exception as e:
            logger.error(f"⚠️ Failed to parse triples: {e}")
        return [], {}


    # ----------------- Insert into Neo4j -----------------
    def insert_triples(self, triples, alias_map=None):
        """
        Insert triples as (h)-[:REL]->(t) edges only.
        Optionally, store aliases for entities for more flexible querying.
        
        alias_map: dict
            Example: {"Elon Musk": ["Elon Reeve Musk", "Elon"]}
        """
        try:
            if not self.driver:
                st.error("❌ Neo4j driver is not initialized. Check URI/user/pass.")
                return

            if not triples:
                st.warning("⚠️ No triples to insert.")
                return

            with self.driver.session() as session:
                for head, relation, tail in triples:
                    safe_rel = text_cleaner(relation).upper().replace(" ", "_")

                    # Merge main entities
                    query = f"""
                        MERGE (h:ENTITY {{name: $head}})
                        MERGE (t:ENTITY {{name: $tail}})
                        MERGE (h)-[:{safe_rel}]->(t)
                    """
                    session.run(query, head=head, tail=tail)

                    # Add aliases if provided
                    if alias_map:
                        for entity, aliases in alias_map.items():
                            for a in aliases:
                                alias_query = """
                                    MATCH (e:ENTITY {name: $entity})
                                    MERGE (alias:ALIAS {name: $alias})
                                    MERGE (alias)-[:ALIAS_OF]->(e)
                                """
                                session.run(alias_query, entity=entity, alias=a)

            st.success(f"✅ Inserted {len(triples)} triples into Neo4j")
            self.visualize_triples()

        except Exception as e:
            st.error(f"❌ Failed to insert triples: {e}")

    # ----------------- Build KG -----------------
    def build_kg(self, user_input: str):
        prompt = f"""
                Extract knowledge triples strictly in JSON list format, including optional aliases:
                [["Entity1", "RELATION", "Entity2", ["Alias1", "Alias2", ...]], ...]

                Rules:
                - Output only a valid Python list of lists. Do not add explanations, notes, or extra text.
                - Each list must contain:
                    1. Entity1 (title case)
                    2. Relation (short, uppercase, noun/verb-like)
                    3. Entity2 (title case or string literal)
                    4. Optional aliases for Entity1 (list of strings; can be empty if none)
                - Examples of relations: "BORN_IN", "PROFESSION", "FOUNDED", "KNOWN_FOR", "HEADQUARTERED".
                - Entity names should be in title case (e.g., "Javed Akhtar", "Apple Inc.").
                - Entity2 can be a literal value (date, number, string) or another entity.
                - Format literal values as strings.
                - Include common variations for aliases, such as first name, last name, or commonly used short names.
                - If the relation is ambiguous, choose the closest meaningful relation from the examples.
                - Avoid generic relations like "is" or "has". Use specific relations that describe the fact.

                Text: "{user_input}"
                """

        response = llm.invoke(prompt)
        raw_triples = response.content.strip()

        triples, alias_map = self.parse_triples(raw_triples)
        self.insert_triples(triples, alias_map)

    # ----------------- Data Visualizer -----------------
    def visualize_triples(self):
        with self.driver.session() as session:
            query = """
            MATCH (h:ENTITY)-[r]->(t:ENTITY)
            RETURN h.name AS head, type(r) AS relation, t.name AS tail
            LIMIT 50
            """
            results = session.run(query)
            triples = [(record["head"], record["relation"], record["tail"]) for record in results]

        # Create Pyvis network
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)

        net.repulsion(
            node_distance=800,   # increase distance between nodes
            central_gravity=0.05,
            spring_length=200,
            spring_strength=0.05,
            damping=0.9
        )

        for h, r, t in triples:
            net.add_node(h, label=h, title=h)
            net.add_node(t, label=t, title=t)
            net.add_edge(h, t, label=r)

        net.set_options("""{
            "edges": {"color": {"inherit": "from"}},
            "nodes": {"size": 15}
        }""")

        # Save HTML into session_state
        net.save_graph("graph.html")
        with open("graph.html", "r", encoding="utf-8") as f:
            st.session_state["graph_html"] = f.read()

        # Render
        # components.html(st.session_state["graph_html"], height=600, width=1000, scrolling=True)

    # ----------------- Query KG -----------------
    def generate_query(self, user_question: str):
        try:
            schema_snippet = self.generate_schema_snippet()
            prompt = f"""
                    You are an expert in generating Cypher queries for Neo4j.

                    Schema:
                        {schema_snippet}

                    Important rules:
                    1. Always match the relationship direction exactly as it exists in the database.
                    2. Questions may use different terms for the same relationship. For example:
                        - "birthplace", "where born", "birth location" → BORN_IN
                        - "job", "occupation", "profession" → PROFESSION
                        - "founded", "established" → FOUNDED or FOUNDED_BY
                    3. If the database stores the full name but the user provides only a partial name (first name, last name, or nickname), first check for a matching ENTITY node with CONTAINS, and if needed check ALIAS nodes. Examples:
                        - MATCH (p:ENTITY) WHERE p.name CONTAINS "Elon" RETURN p.name AS result
                        - MATCH (a:ALIAS)-[:ALIAS_OF]->(p:ENTITY) WHERE a.name CONTAINS "Elon" RETURN p.name AS result
                    4. Family relationships (PARENT, CHILD, SPOUSE, SIBLING) always connect directly between people nodes.
                    5. Always alias return values as `AS result`.
                    6. Use the shortest valid query possible while respecting these rules.
                    7. When generating Cypher queries, always:
                        - Normalize entity names.
                        - Replace en dash (–), em dash (—), and minus signs (−) with a standard ASCII hyphen (-).
                        - Remove unnecessary whitespace.


                    Examples:
                    Q: Who founded Apple Inc.?
                    A: MATCH (a:ENTITY {{name:"Apple Inc."}})-[:FOUNDED_BY]->(f:ENTITY) RETURN f.name AS result

                    Q: Who are Elon Musk's parents?
                    A: MATCH (p:ENTITY {{name:"Elon Reeve Musk"}})-[:PARENT]->(c:ENTITY) RETURN c.name AS result

                    Q: Where was Javed Akhtar born?
                    A: MATCH (j:ENTITY {{name:"Javed Akhtar"}})-[:BORN_IN]->(b:ENTITY) RETURN b.name AS result

                    Q: Who acted in Screamers?
                    A: MATCH (a:ENTITY)-[:ACTED_IN]->(m:ENTITY {{name:"Screamers"}}) RETURN a.name AS result

                    Now generate ONLY the Cypher query for the following question.
                    Question: "{user_question}"
                    """

            response = llm.invoke(prompt)
            cypher_query = clean_cypher(response.content.strip())

            st.write("📝 Generated Cypher:", cypher_query)

            return self.query_kg(cypher_query)

        except Exception as e:
            st.error(f"❌ Query generation failed: {e}")
            return []

    def query_kg(self, cypher_query: str):
        with self.driver.session() as session:
            try:
                results = session.run(cypher_query)
                data = [dict(record) for record in results]
                if not data:
                    st.info("⚠️ No results found. Try including keywords from the Schema Snippet in your query.")
                    return []

                # Show only values
                values = [list(record.values())[0] for record in data]
                st.success("✅ Query Results:")
                st.write(values)

                return values
            except Exception as e:
                st.error(f"❌ KG query failed: {e}")
                return []
    
    def reset_kg(self):
        """Delete all nodes and relationships in Neo4j"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("🗑️ Deleted all existing triples from Neo4j")
            st.info("🗑️ Knowledge Graph reset (all old triples deleted)")
        except Exception as e:
            logger.error(f"⚠️ Failed to reset KG: {e}")
            st.error(f"⚠️ Failed to reset KG: {e}")
    
    # ----------------- Schema Introspection -----------------
    def get_labels_and_properties(self):
        """Fetch all node labels and their properties from Neo4j."""
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return {}

        query = """
        CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes
        RETURN nodeLabels, collect({property: propertyName, types: propertyTypes}) as properties
        """
        with self.driver.session() as session:
            result = session.run(query)
            schema = {}
            for record in result:
                labels = tuple(record["nodeLabels"])
                props = record["properties"]
                schema[labels] = props
            return schema

    def get_relationship_types(self):
        """Fetch all relationship types from Neo4j."""
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return []

        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as relTypes"
        with self.driver.session() as session:
            result = session.run(query)
            return result.single()["relTypes"]

    def generate_schema_snippet(self):
        """Generate a prompt-ready text snippet for LLM using current schema."""
        labels_props = self.get_labels_and_properties()
        rel_types = self.get_relationship_types()

        snippet = "Schema info:\n"
        for labels, props in labels_props.items():
            snippet += f"- Node Label(s): {labels}\n"
            snippet += "  Properties:\n"
            for p in props:
                snippet += f"    - {p['property']} ({', '.join(p['types'])})\n"
        snippet += "\nRelationship Types:\n"
        for r in rel_types:
            snippet += f"- {r}\n"
        return snippet