import sqlite3

def create_database():

    conn = sqlite3.connect("maintenance.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            TP2 REAL,
            TP3 REAL,
            H1 REAL,
            Motor_current REAL,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


create_database()