# Funktioner för att skapa databasen och lägga till data i den.

import sqlite3

DB_FILE = "iskartor.db"

# Skapar databasen.
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS icemap_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area TEXT,
            last_modified TEXT,
            expires TEXT,
            filepath TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_metadata(area, last_modified, expires, filepath):
    """
    Sparar metadatan i en databas.
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO icemap_metadata (area, last_modified, expires, filepath) VALUES (?, ?, ?, ?)",
                (area, last_modified.isoformat() if last_modified else None,
                 expires.isoformat() if expires else None,
                 filepath))
    conn.commit()
    conn.close()