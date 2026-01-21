import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, project_root)

import psycopg
from src.core import config


def _get_db_params() -> dict:
    return {
        "host": config.POSTGRES_HOST,
        "port": config.POSTGRES_PORT,
        "user": config.POSTGRES_USER,
        "password": config.POSTGRES_PASSWORD,
        "dbname": config.POSTGRES_DB,
    }


def check_table_exists(table_name: str) -> bool:
    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE  table_schema = 'public'
            AND    table_name   = %s
        );
    """

    try:
        with psycopg.connect(**_get_db_params()) as conn:
            result = conn.execute(check_query, (table_name,)).fetchone()
            return result[0] if result else False

    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False
