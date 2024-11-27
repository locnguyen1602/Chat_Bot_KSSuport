from typing import Optional, Dict, Any
from langchain_community.utilities import SQLDatabase
import pandas as pd
from .config import settings


class DatabaseManager:
    """Service to handle database related issues"""

    def __init__(self):
        self.db = None

    def connect_database(self, database_name: str) -> None:
        """Establish a connection to the MySQL database

        Args:
            database_name (str): Database name to connect to
        """
        try:
            # Replace placeholder in connection string with actual database name
            conn_string = settings.DB_CONNECTION_STRING.format(database_name)
            self.db = SQLDatabase.from_uri(conn_string)
            print(f"Connected to database {database_name} successfully")
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            raise

    def get_data(self, query: str, params: tuple = None) -> Optional[pd.DataFrame]:
        """Get data from database using SQL query with optional parameters

        Args:
            query (str): SQL query string
            params (tuple, optional): Query parameters

        Returns:
            Optional[pd.DataFrame]: Query results or None if error occurs
        """

        if self.db is None:
            print("Error: Database connection is not established")
            return None

        try:
            engine = self.db._engine

            # Execute query with or without parameters
            if params is not None:
                df = pd.read_sql_query(sql=query, con=engine, params=params)
            else:
                df = pd.read_sql_query(sql=query, con=engine)

            return df

        except Exception as e:
            print(f"Error in get_data: {str(e)}")
            # Print the full error traceback for debugging
            import traceback

            print("Full error traceback:")
            print(traceback.format_exc())
            return None

        finally:
            print("=== Query execution completed ===\n")


# Initialize DatabaseManager
db_manager = DatabaseManager()
