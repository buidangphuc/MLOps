from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator  # Updated from SimpleHttpOperator
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
import json
import logging

## Define the DAG
with DAG(
    dag_id='nasa_apod_postgres',
    start_date=days_ago(1),
    schedule_interval='@once',
    catchup=False,
    # Thêm các thông tin hiển thị trên UI
    tags=['nasa', 'postgres', 'etl'],
    description='ETL pipeline to extract NASA APOD data and load into PostgreSQL',
    doc_md="""
    # NASA APOD ETL Pipeline
    
    This DAG performs the following operations:
    1. Creates a table in PostgreSQL if it doesn't exist
    2. Extracts data from NASA's Astronomy Picture of the Day API
    3. Transforms the data to select required fields
    4. Loads the data into PostgreSQL
    5. Verifies the data by displaying statistics
    
    **Note:** Requires NASA API connection and PostgreSQL connection to be configured in Airflow
    """
) as dag:
    ## step 1: Create the table if it doesnt exists

    @task
    def create_table():
        ## initialize the Postgreshook
        postgres_hook=PostgresHook(postgres_conn_id="my_postgres_connection")

        ## SQL query to create the table
        create_table_query="""
        CREATE TABLE IF NOT EXISTS apod_data (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255),
            explanation TEXT,
            url TEXT,
            date DATE,
            media_type VARCHAR(50)
        );


        """
        ## Execute the table creation query
        postgres_hook.run(create_table_query)


    ## Step 2: Extract the NASA API Data(APOD)-Astronomy Picture of the Day[Extract pipeline]
    ## https://api.nasa.gov/planetary/apod?api_key=7BbRvxo8uuzas9U3ho1RwHQQCkZIZtJojRIr293q
    extract_apod = HttpOperator(  # Updated from SimpleHttpOperator
        task_id='extract_apod',
        http_conn_id='nasa_api',  ## Connection ID Defined In Airflow For NASA API
        endpoint='planetary/apod', ## NASA API enpoint for APOD
        method='GET',
        data={"api_key":"{{ conn.nasa_api.extra_dejson.api_key}}"}, ## USe the API Key from the connection
        response_filter=lambda response:response.json(), ## Convert response to json
    )

    

    ## Step 3: Transform the data(Pick the information that i need to save)
    @task
    def transform_apod_data(response):
        apod_data={
            'title': response.get('title', ''),
            'explanation': response.get('explanation', ''),
            'url': response.get('url', ''),
            'date': response.get('date', ''),
            'media_type': response.get('media_type', '')

        }
        return apod_data


    ## step 4:  Load the data into Postgres SQL
    @task
    def load_data_to_postgres(apod_data):
        ## Initialize the PostgresHook
        postgres_hook=PostgresHook(postgres_conn_id='my_postgres_connection')

        ## Define the SQL Insert Query

        insert_query = """
        INSERT INTO apod_data (title, explanation, url, date, media_type)
        VALUES (%s, %s, %s, %s, %s);
        """

        ## Execute the SQL Query

        postgres_hook.run(insert_query,parameters=(
            apod_data['title'],
            apod_data['explanation'],
            apod_data['url'],
            apod_data['date'],
            apod_data['media_type']


        ))

    ## step 5: Verify the data with a count and log results
    @task
    def verify_data():
        postgres_hook = PostgresHook(postgres_conn_id='my_postgres_connection')
        
        # Count records and get the latest entry
        count_query = "SELECT COUNT(*) FROM apod_data;"
        latest_query = """
        SELECT title, date FROM apod_data
        ORDER BY date DESC LIMIT 1;
        """
        
        count_result = postgres_hook.get_first(count_query)[0]
        latest_entry = postgres_hook.get_first(latest_query)
        
        logging.info(f"Total records in apod_data table: {count_result}")
        if latest_entry:
            logging.info(f"Latest entry: {latest_entry[0]} from {latest_entry[1]}")
        
        return {"record_count": count_result, "latest_record": latest_entry[0] if latest_entry else None}

    ## step 6: Define the task dependencies
    ## Extract
    create_table() >> extract_apod  ## Ensure the table is create befor extraction
    api_response = extract_apod.output
    ## Transform
    transformed_data = transform_apod_data(api_response)
    ## Load
    load_task = load_data_to_postgres(transformed_data)
    ## Verify
    verification = verify_data()
    
    # Updated task dependencies to include verification
    load_task >> verification