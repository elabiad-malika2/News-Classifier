from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
import joblib

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1)
}

dag = DAG(
    "news_pipeline_existing_model",
    default_args=default_args,
    description="Pipeline AG News → Cleaning → Embeddings → ChromaDB avec modèle existant",
    schedule_interval=None,
    catchup=False,
)

# --- Charger modèle existant ---
model_path = r"C:\Users\elabi\OneDrive\Desktop\New Classifier\modele\news_classifier_logreg_modele.pkl"
encoder_path = r"C:\Users\elabi\OneDrive\Desktop\New Classifier\modele\label_encoder.pkl"
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# --- Tâches Python ---

def load_data():
    ds = load_dataset("SetFit/ag_news")
    df_train = pd.DataFrame(ds['train'])
    df_test = pd.DataFrame(ds['test'])
    df_train.to_csv("../tmp/train.csv", index=False)
    df_test.to_csv("../tmp/test.csv", index=False)

def clean_text():
    df_train = pd.read_csv("/tmp/train.csv")
    df_test = pd.read_csv("/tmp/test.csv")
    df_train['clean_text'] = df_train['text'].str.replace(r'\W+', ' ', regex=True).str.lower()
    df_test['clean_text'] = df_test['text'].str.replace(r'\W+', ' ', regex=True).str.lower()
    df_train.to_csv("/tmp/train_clean.csv", index=False)
    df_test.to_csv("/tmp/test_clean.csv", index=False)

def embeddings_and_store():
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path="/tmp/chroma_db")
    collection_train = client.get_or_create_collection(name="news_train")
    collection_test = client.get_or_create_collection(name="news_test")

    for split in ["train", "test"]:
        df = pd.read_csv(f"/tmp/{split}_clean.csv")
        # Embeddings
        df['embedding'] = df['clean_text'].apply(lambda x: embedder.encode(x).tolist())

        # Ajouter dans ChromaDB
        ids = df.index.astype(str).tolist()
        embeddings = df['embedding'].tolist()
        metadatas = df[['label', 'clean_text']].to_dict('records')

        if split == "train":
            collection_train.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        else:
            collection_test.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

# --- Définition des tâches ---
t1 = PythonOperator(task_id="load_data", python_callable=load_data, dag=dag)
t2 = PythonOperator(task_id="clean_text", python_callable=clean_text, dag=dag)
t3 = PythonOperator(task_id="embeddings_and_store", python_callable=embeddings_and_store, dag=dag)

# --- Orchestration ---
t1 >> t2 >> t3
