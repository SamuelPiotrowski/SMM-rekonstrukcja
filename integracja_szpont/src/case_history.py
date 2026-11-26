"""Local archive of uploaded meshes and their reconstruction results."""

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone


class CaseHistory:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS folders (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, normalized TEXT UNIQUE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, folder TEXT REFERENCES folders(id),
                    created TEXT NOT NULL, mixed INTEGER NOT NULL, result TEXT, error TEXT, anchor INTEGER
                );
                CREATE TABLE IF NOT EXISTS scans (
                    case_id TEXT REFERENCES cases(id), number INTEGER, name TEXT NOT NULL, data BLOB NOT NULL,
                    PRIMARY KEY (case_id, number)
                );
            """)
            db.execute("BEGIN IMMEDIATE")
            if not any(row["name"] == "anchor" for row in db.execute("PRAGMA table_info(cases)")):
                db.execute("ALTER TABLE cases ADD COLUMN anchor INTEGER")

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            yield db

    def create_folder(self, name):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
            raise ValueError("Podaj nazwę folderu od 1 do 60 znaków.")
        name = name.strip()
        key = uuid.uuid4().hex
        try:
            with self._connect() as db:
                db.execute("INSERT INTO folders VALUES (?, ?, ?)", (key, name, name.casefold()))
        except sqlite3.IntegrityError:
            raise ValueError("Folder o tej nazwie już istnieje.") from None
        return {"id": key, "name": name}

    def folder_id(self, db, key):
        if key in (None, ""):
            return None
        if not isinstance(key, str) or not db.execute("SELECT 1 FROM folders WHERE id = ?", (key,)).fetchone():
            raise ValueError("Wybierz istniejący folder.")
        return key

    def add(self, key, title, folder, mixed, files, anchor=None):
        with self._connect() as db:
            db.execute("INSERT INTO cases (id, title, folder, created, mixed, anchor) VALUES (?, ?, ?, ?, ?, ?)",
                       (key, title, self.folder_id(db, folder), datetime.now(timezone.utc).isoformat(), mixed, anchor))
            db.executemany("INSERT INTO scans VALUES (?, ?, ?, ?)",
                           ((key, index, name, data) for index, (name, data) in enumerate(files, 1)))

    def move(self, key, folder):
        with self._connect() as db:
            updated = db.execute("UPDATE cases SET folder = ? WHERE id = ?", (self.folder_id(db, folder), key))
            return bool(updated.rowcount)

    def list(self):
        with self._connect() as db:
            db.execute("BEGIN")
            folders = [dict(row) for row in db.execute("SELECT id, name FROM folders ORDER BY normalized")]
            cases = {row["id"]: dict(row, files=[]) for row in db.execute("""
                SELECT id, title, folder, created, result IS NOT NULL AS done, error
                FROM cases ORDER BY created DESC
            """)}
            for row in db.execute("SELECT case_id, number, name FROM scans ORDER BY number"):
                cases[row["case_id"]]["files"].append({"number": row["number"], "name": row["name"]})
        return {"folders": folders, "cases": list(cases.values())}

    def inputs(self, key):
        with self._connect() as db:
            case = db.execute("SELECT title, mixed, anchor FROM cases WHERE id = ?", (key,)).fetchone()
            files = db.execute("SELECT name, data FROM scans WHERE case_id = ? ORDER BY number", (key,))
            return case["title"], bool(case["mixed"]), [tuple(row) for row in files], case["anchor"]

    def scan(self, key, number):
        with self._connect() as db:
            row = db.execute("SELECT name, data FROM scans WHERE case_id = ? AND number = ?", (key, number)).fetchone()
            return tuple(row) if row else None

    def result(self, key):
        with self._connect() as db:
            return db.execute("""
                SELECT result, error, EXISTS (SELECT 1 FROM scans WHERE case_id = cases.id AND number = 2) AS can_solve
                FROM cases WHERE id = ?
            """, (key,)).fetchone()

    def finish(self, key, result):
        content = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as db:
            db.execute("UPDATE cases SET result = ?, error = NULL WHERE id = ? AND result IS NULL", (content, key))

    def fail(self, key, message):
        with self._connect() as db:
            db.execute("UPDATE cases SET error = ? WHERE id = ? AND result IS NULL", (message, key))
