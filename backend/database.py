import sqlite3
import json
import os

# Use __file__-relative paths so files are always found regardless of CWD.
# On Vercel, the filesystem is read-only except for /tmp, so DB_PATH
# can be overridden via the DB_PATH env var (e.g. /tmp/blooms.db).
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.environ.get("DB_PATH", os.path.join(_HERE, "blooms.db"))
SEED_FILE = os.path.join(_HERE, "blooms_verbs.json")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blooms_verbs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verb TEXT UNIQUE NOT NULL,
            taxonomy_level TEXT NOT NULL,
            level_weight INTEGER NOT NULL
        )
    ''')

    # Load seed data
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, 'r') as f:
            verbs = json.load(f)
            for v in verbs:
                cursor.execute('''
                    INSERT OR REPLACE INTO blooms_verbs (verb, taxonomy_level, level_weight)
                    VALUES (?, ?, ?)
                ''', (v['verb'], v['taxonomy_level'], v['level_weight']))
    conn.commit()
    conn.close()


def get_verb_info(verb: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT taxonomy_level, level_weight FROM blooms_verbs WHERE verb = ?', (verb.lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"taxonomy_level": row[0], "level_weight": row[1]}
    return None


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
