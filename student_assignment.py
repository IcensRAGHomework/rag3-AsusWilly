import datetime
import chromadb
import pandas as pd
import time
import datetime

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file = "COA_OpenData.csv"

def generate_hw01():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chromadb.PersistentClient(path="chroma.sqlite3").get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    try:
        df = pd.read_csv(csv_file)
        if collection.count() != df.shape[0]:
            for index, row in df.iterrows():
                timestamp = int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())
                collection.add(
                    ids=[str(row['ID'])],
                    documents=row['HostWords'],
                    metadatas=[{
                        'name': row['Name'],
                        'type': row['Type'],
                        'address': row['Address'],
                        'tel': row['Tel'],
                        'city': row['City'],
                        'town': row['Town'],
                        'date': timestamp }]
                )
    except:
        return collection

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    result = collection.query(
        query_texts = [question],
        n_results = 10,
        where={"$and": [
            {'city': {"$in": city}}, 
            {'type': {"$in": store_type}},
            {'date': {"$gte": start_timestamp}},
            {'date': {"$lte": end_timestamp}}
            ]},
        include=["metadatas", "distances"],
    )
    if result.get('metadatas') and result.get('distances'):
        return [
            metadata['name']
            for metadata, distance in zip(result['metadatas'][0], result['distances'][0])
            if distance < 0.2
        ]
    return []

def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()
    results = collection.query(
        query_texts=[store_name],
        n_results = 10,
        where={'name': {'$eq': store_name}}
    )
    new_metadata = results['metadatas'][0][0]
    new_metadata['new_store_name'] = new_store_name
    
    collection.delete(ids=[results['ids'][0][0]])
    collection.add(
        ids=[results['ids'][0][0]],  
        documents=results['documents'][0],
        metadatas=[new_metadata]  
    )
    result = collection.query(
        query_texts=[question],
        where={'$and': [
            {'city': {'$in': city}}, 
            {'type': {'$in': store_type}},
            ]},
        include=["metadatas", "distances"],
    )

    return [metadata['new_store_name']
            if metadata.get('new_store_name')
            else metadata['name']
            for metadata, distance in zip(result['metadatas'][0], result['distances'][0])
            if distance < 0.2]

    if result.get('metadatas') and result.get('distances'):
        return [
            metadata.get('new_store_name', metadata.get('name', 'The store cannot be found!'))
            for metadata, distance in zip(result['metadatas'][0], result['distances'][0])
            if distance < 0.2
        ]
    return []

def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    return collection


print(generate_hw01().count())
print(generate_hw02(question="我想要找有關茶餐點的店家",
                    city=["宜蘭縣", "新北市"],
                    store_type=["美食"],
                    start_date=datetime.datetime(2024, 4, 1),
                    end_date=datetime.datetime(2024, 5, 1)))
print(generate_hw03(question="我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵",
                    store_name="耄饕客棧",
                    new_store_name="田媽媽（耄饕客棧）",
                    city=["南投縣"],
                    store_type=["美食"]))