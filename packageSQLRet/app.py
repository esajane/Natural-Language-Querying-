import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql.base import SQLDatabaseChain

def create_database_engine(user, password, host, port, db_name, db_type='mysql'):
    """
    Create a database engine using SQLAlchemy.
    """
    db_url = f"{db_type}+pymysql://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(db_url)

def setup_google_genai(api_key, model="models/text-bison-001", temperature=0.1):
    """
    Initialize and return a Google Generative AI model.
    """
    return GoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)

def get_schema(engine):
    """
    Retrieve and return the database schema: tables and their column details.
    """
    inspector = inspect(engine)
    schema_info = {}
    for table_name in inspector.get_table_names():
        schema_info[table_name] = [col['name'] for col in inspector.get_columns(table_name)]
    return schema_info

st.title("Database Connection and Query App")

with st.expander("Setup Database Connection"):
    db_type = st.selectbox("Select Database Type", ["mysql", "postgresql"], index=0)
    db_user = st.text_input("Database User")
    db_password = st.text_input("Database Password", type="password")
    db_host = st.text_input("Database Host")
    db_port = st.number_input("Database Port", min_value=1, max_value=65535, value=3306)
    db_name = st.text_input("Database Name")
    api_key = st.text_input("Google API Key", type="password")

    if st.button("Setup Database and Model"):
        try:
            engine = create_database_engine(db_user, db_password, db_host, db_port, db_name, db_type)
            db = SQLDatabase(engine)
            schema_info = get_schema(engine)
            st.session_state['schema_info'] = schema_info
            llm = setup_google_genai(api_key)
            db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
            st.session_state['db_chain'] = db_chain
            st.success("Database and model setup successfully!")

        except Exception as e:
            st.error(f"Failed to setup: {str(e)}")

with st.expander("Run Query"):
    query = st.text_input("Enter your query in natural language")
    if query and st.button("Run Query"):
        if 'db_chain' in st.session_state:
            try:
                result = st.session_state['db_chain'].invoke(query)
                if isinstance(result, dict) and 'result' in result:
                    items = result['result'].split(', ')  
                    data = pd.DataFrame(items, columns=['Event'])  
                else:
                    data = pd.DataFrame([{'Error': 'Unexpected result format'}])
                
                st.table(data)  
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
        else:
            st.error("Please setup the database and model first.")


st.caption("Powered by Streamlit and Google Generative AI")
