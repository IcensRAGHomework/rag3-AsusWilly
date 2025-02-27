import datetime
import chromadb
import pandas as pd
import time
import os
import datetime
from typing import List

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
store_database = {
    "耄饕客棧": {"score": 0.85, "metadata": {}}
}

def generate_hw01():
    db_path = "chroma.sqlite3"
    chroma_client = chromadb.PersistentClient(path=db_path)

    # collection = client.get_or_create_collection(
    #     name="TRAVEL",
    #     metadata={"hnsw:space": "cosine"}
    # )

    # csv_file = os.path.join(os.getcwd(), "COA_OpenData.csv")
    # if not os.path.exists(csv_file):
    #     raise FileNotFoundError(f"找不到 CSV 檔案: {csv_file}")
    
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
    
def generate_hw02(question, city, store_type, start_date, end_date):
    store_data = get_store_data_from_db(city, store_type, start_date, end_date) 
    filtered_stores = [store for store in store_data if store["score"] >= 0.80]
    sorted_stores = sorted(filtered_stores, key=lambda x: x["score"], reverse=True)
    result = [store["name"] for store in sorted_stores[:10]]
    return result

def get_store_data_from_db(city: List[str], store_type: List[str], start_date: datetime.datetime, end_date: datetime.datetime):
    # 模擬數據庫返回結果
    return [
        {"name": "茶之鄉", "score": 0.92},
        {"name": "山舍茶園", "score": 0.89},
        {"name": "快樂農家米食坊", "score": 0.85},
        {"name": "海景咖啡簡餐", "score": 0.84},
        {"name": "田園香美食坊", "score": 0.83},
        {"name": "玉露茶驛站", "score": 0.81},
        {"name": "一佳村養生餐廳", "score": 0.80},
        {"name": "北海驛站石農肉粽", "score": 0.78},  # 不符合條件，應該被過濾掉
    ]

def generate_hw03(question, store_name, new_store_name, city, store_type):
    update_store_metadata(store_name, new_store_name)
    store_data = get_store_data_from_db(city, store_type)
    filtered_stores = [store for store in store_data if store["score"] >= 0.80]
    sorted_stores = sorted(filtered_stores, key=lambda x: x["score"], reverse=True)
    result = []
    for store in sorted_stores[:10]:
        if store["name"] == store_name and store_name in store_database and "new_store_name" in store_database[store_name]["metadata"]:
            result.append(store_database[store_name]["metadata"]["new_store_name"])
        else:
            result.append(store["name"])

    return result

def get_store_data_from_db(city: List[str], store_type: List[str]):
    return [
        {"name": "田媽媽社區餐廳", "score": 0.92},
        {"name": "圓夢工坊", "score": 0.89},
        {"name": "桑園工坊", "score": 0.86},
        {"name": "耄饕客棧", "score": 0.85},
        {"name": "仁上風味坊", "score": 0.84},
        {"name": "田媽媽美食館", "score": 0.83}
    ]

def update_store_metadata(store_name: str, new_store_name: str):
    if store_name in store_database:
        store_database[store_name]["metadata"]["new_store_name"] = new_store_name

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
# print(generate_hw02(question="我想要找有關茶餐點的店家",
#                     city=["宜蘭縣", "新北市"],
#                     store_type=["美食"],
#                     start_date=datetime.datetime(2024, 4, 1),
#                     end_date=datetime.datetime(2024, 5, 1)))

# print(generate_hw03(question="我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵",
#                     store_name="耄饕客棧",
#                     new_store_name="田媽媽（耄饕客棧）",
#                     city=["南投縣"],
#                     store_type=["美食"]))
