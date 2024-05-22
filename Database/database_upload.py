from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv
from io import StringIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("upload_log.log"),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

def upload_processed_data_to_azure(file_content, env_path):
    """
    Upload processed data from a CSV file to an Azure SQL database.

    Args:
        csv_file_path (str): Path to the CSV file containing processed data.
        table_name (str): Name of the table to upload the data to.
        
    """

    logger.info("Starting upload process")
    logger.info("Loading environment variables")
     # Load environment variables from the specified .env file
    load_dotenv(env_path)

    # Retrieve database connection details from environment variables
    server = os.getenv("DB_SERVER1")
    database = os.getenv("DB_DATABASE1")
    username = os.getenv("DB_USERNAME1")
    password = os.getenv("DB_PASSWORD1")
    driver = "ODBC Driver 18 for SQL Server"
    
    # Connection string
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

    logger.info(f"Database details: Server={server}, Database={database}, User={username} ")
    # Use the CSV file name as the table name
    table_name = os.path.splitext(os.path.basename(csv_file_path))[0]   
    
    # Create SQLAlchemy engine
    try:
        engine = create_engine(connection_string)
        # Test the connection
        with engine.connect() as conn:
            logger.info("Connection to database successful")

        # Load the processed dataset
        dataset = pd.read_csv(StringIO(file_content))
        logger.info("CSV file content loaded into DataFrame")

        # Export dataset to Azure SQL Database
        dataset.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"Dataset has been successfully exported to the {table_name} table")

    except Exception as e:
        print(f"An error occurred: {e}")

# Define the file paths and table name

if __name__ == "__main__":
    logger.info("Script started")

    # Define paths to read from the memory
    csv_file_path = "D:/BUAS/Year 2/Block 2D/Block D/Database/Upload/test_output.csv"  # Update this with the correct path to your CSV file
    env_path =os.path.abspath('D:/BUAS/Year 2/Block 2D/Block-D-Personal/Programming/.env.txt')
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()  # Read the file content as a string
            logger.info(f"File {csv_file_path} read successfully")
    except Exception as e:
        logger.error(f"Failed to read file {csv_file_path}: {e}")
        raise   

    upload_processed_data_to_azure(file_content, env_path)
    logger.info("Script finished")
