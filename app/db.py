from contextlib import contextmanager
import pymysql
from pymysql.cursors import DictCursor
from .config import settings


@contextmanager
def db_connection():
    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_username,
        password=settings.db_password,
        database=settings.db_database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(sql: str, params=None):
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or {})
            return cursor.fetchall()


def fetch_one(sql: str, params=None):
    rows = fetch_all(sql, params)
    return rows[0] if rows else None