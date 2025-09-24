# src/common/database/create_all_tables.py
from .MQSDBConnector import MQSDBConnector
from .schemaDefinitions import SchemaDefinitions


def main():
    """
    Main function to initialize the database by creating all tables.
    """
    print("Attempting to connect to the database and create tables...")
    
    try:
        # 1. Create an instance of the SchemaDefinitions class.
        #    This will also initialize the database connection via MQSDBConnector.
        schema_manager = SchemaDefinitions()

        # 2. Call the method to execute the CREATE TABLE statements.
        schema_manager.create_all_tables()

        print("\nScript finished.")

    except ImportError:
        print("Error: Could not import SchemaDefinitions.")
        print("Please ensure this script is in the same directory as 'schema_definitions.py'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()