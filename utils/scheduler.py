from apscheduler.schedulers.background import BackgroundScheduler
from neo4j import GraphDatabase
import atexit
from configs.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import streamlit as st
import threading
import time

if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in config")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def ping_neo4j():
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        st.success("✅ Neo4j is online")
    except Exception as e:
        st.error("❌ Neo4j connection failed: {e}")
        # here you can send email/slack alert if needed

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(ping_neo4j, "interval", minutes=30)
    scheduler.start()

    # Ensure scheduler stops when app exits
    atexit.register(lambda: scheduler.shutdown())

def keep_neo4j_alive(interval_minutes: int = 30):
    """Continuously ping the Neo4j DB to prevent it from sleeping."""
    def ping():
        while True:
            try:
                with driver.session() as session:
                    session.run("RETURN 1 AS ping")
                print("✅ Neo4j keep-alive ping sent.")
            except Exception as e:
                print(f"⚠️ Neo4j keep-alive failed: {e}")
            time.sleep(interval_minutes * 60)
    
    thread = threading.Thread(target=ping, daemon=True)
    thread.start()
    return driver