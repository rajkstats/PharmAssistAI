# Set up a Python virtual environment and activate it:
# python3 -m venv myenv
# source myenv/bin/activate

# Run the FastAPI application with live reload for development:
# uvicorn main:app --reload

# Run the FastAPI application in the background for production:
# nohup uvicorn main:app --host 0.0.0.0 --port 8000 &

# Check which process is using a specific port (8000 in this case):
# lsof -i :8000

# Terminate a process using a specific port by its PID (e.g., PID 2540):
# kill -9 2540

# Example of a POST request from Postman or any other HTTP client:
# This request indexes data from a specified URL:
# You can call this endpoint by sending a POST request to:
# http://your_server_url/index_fda_drugs?url=https://download.open.fda.gov/drug/label/drug-label-0001-of-0012.json.zip
# where the URL is passed as a query parameter.

# Examples for testing the endpoint locally using curl:
# Local URL testing with curl:
# curl -X POST "http://127.0.0.1:8000/index_fda_drugs?url=https://download.open.fda.gov/drug/label/drug-label-0001-of-0012.json.zip"

import asyncio
import time
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
import pandas as pd
import zipfile
import io
import requests
import json
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import DataFrameLoader
import uuid
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()

# Load environment variables from a .env file
load_dotenv()

# Set up Qdrant client and embedding model
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = os.environ.get("QDRANT_CLUSTER_URL")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
client = AsyncQdrantClient(QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)

async def create_collection():
    try:
        collection_info = await client.get_collection(collection_name="fda_drugs")
        print(f"Collection 'fda_drugs' already exists.")
    except Exception as e:
        print(f"Collection 'fda_drugs' does not exist. Creating...")
        collection_info = await client.create_collection(
            collection_name="fda_drugs",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        print(f"Collection 'fda_drugs' created: {collection_info}")

async def index_batch(batch_docs, metadata_fields):
    points = []
    for doc in batch_docs:
        try:
            vector = embedding_model.embed_query(doc.page_content)
            if vector is not None:
                payload = {field: doc.metadata.get(field, '') for field in metadata_fields}
                payload["page_content"] = doc.page_content
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload=payload,
                    vector=vector,
                ))
        except Exception as e:
            print(f"Failed to index document: {e}")
    
    if points:
        try:
            response = await client.upsert(
                collection_name="fda_drugs",
                points=points,
            )
            return len(points)
        except Exception as e:
            print(f"Failed to upsert batch: {e}")
    
    return 0

@app.post("/index_fda_drugs")
async def index_fda_drugs(url: str = Query(..., description="URL of the ZIP file to index")):
    try:
        start_time = time.time()  # Start timing

        # Create or recreate the collection
        await create_collection()
        
        # Download and load data
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        json_file = zip_file.open(zip_file.namelist()[0])
        data = json.load(json_file)
        df = pd.json_normalize(data['results'])
        selected_drugs = df
        
        # Define metadata fields to include
        metadata_fields = ['openfda.brand_name', 'openfda.generic_name', 'openfda.manufacturer_name', 'openfda.product_type',
                        'openfda.route', 'openfda.substance_name', 'openfda.rxcui', 'openfda.spl_id', 'openfda.package_ndc']
        
        # Fill NaN values with empty strings
        selected_drugs[metadata_fields] = selected_drugs[metadata_fields].fillna('')
        
        # Define text fields to index
        text_fields = ['description', 'indications_and_usage', 'contraindications', 'warnings', 'adverse_reactions', 'dosage_and_administration']
        
        # Fill NaN values with empty strings and concatenate text fields
        selected_drugs[text_fields] = selected_drugs[text_fields].fillna('')
        selected_drugs['page_content'] = selected_drugs[text_fields].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        
        # Create document loader and load drug documents
        loader = DataFrameLoader(selected_drugs, page_content_column='page_content')
        drug_docs = loader.load()
        
        # Update metadata for each document
        for doc, row in zip(drug_docs, selected_drugs.to_dict(orient='records')):
            metadata = {}
            for field in metadata_fields:
                value = row.get(field)
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value if pd.notna(v))
                elif pd.isna(value):
                    value = 'Not Available'
                metadata[field] = value
            doc.metadata = metadata
        
        # Split drug documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_drug_docs = text_splitter.split_documents(drug_docs)
        total_docs = len(split_drug_docs)  # Get the total number of split documents
        
        # Index documents in batches
        batch_size = 100
        indexed_count = 0
        for i in range(0, total_docs, batch_size):
            batch_docs = split_drug_docs[i:i+batch_size]
            batch_count = await index_batch(batch_docs, metadata_fields)
            indexed_count += batch_count
            print(f"Indexed {indexed_count} / {total_docs} documents")
        
        remaining = total_docs - indexed_count
        print(f"Indexing completed. Indexed {indexed_count} / {total_docs}, Remaining: {remaining}")
        
        end_time = time.time()  # End timing
        total_time = end_time - start_time
        print(f"Total time taken to index: {total_time:.2f} seconds")
        
        return {"message": "Indexing completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))