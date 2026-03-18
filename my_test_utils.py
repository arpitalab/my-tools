import os
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
from tqdm import tqdm
import zipfile

xml_folder = "/Users/sraghava/downloaded_xmls" 

def extract_zip(zip_path, extract_folder):
    try:
        print(f"Extracting {zip_path}...")
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
        xml_files = [os.path.join(extract_folder, f) for f in os.listdir(extract_folder) if f.endswith(".xml")]
        print(f"Extracted {len(xml_files)} XML files from {zip_path}")
        return xml_files
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return []

def extract_abstract(xml_path):
    try:
        tree = ET.parse(xml_path)
        abstract = tree.find(".//AbstractNarration")  # Adjust tag if different
        return abstract.text.strip() if abstract is not None and abstract.text else None
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

downloaded_zips = os.listdir(xml_folder)
#print("Extracting XMLs from ZIPs...")
all_xml_files = []
for zip_file in tqdm(downloaded_zips, desc="Extracting ZIPs"):
    xml_files = extract_zip(os.path.join(xml_folder,zip_file), xml_folder+'/'+zip_file+'_xml')
    all_xml_files.extend(xml_files)

print(f"Total XML files extracted: {len(all_xml_files)}")

abstracts = []
for xml_file in tqdm(all_xml_files, desc="Extracting Abstracts"):
    abstract = extract_abstract(xml_file)
    if abstract:
        abstracts.append(abstract)

from sentence_transformers import SentenceTransformer 
import nltk
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')).union(['and', 'in', 'the'])

def preprocess_text(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

texts = [preprocess_text(doc) for doc in abstracts]

embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(texts,show_progress_bar=True)


umap_model = UMAP(n_components=5,min_dist=0.0,metric='cosine')

reduced_embeddings = umap_model.fit_transform(embeddings)

hdbscan_model = HDBSCAN(
    min_cluster_size = 50, metric="euclidean",cluster_selection_method="eom"
).fit(reduced_embeddings)

clusters = hdbscan_model.labels_

cluster = 10
for index in np.where(clusters==cluster)[0][:3]:
    print(texts[index][:300]+"...\n")

reduced_embeddings = UMAP(
    n_components = 2, min_dist = 0.0, metric="cosine"
).fit_transform(embeddings)

df = pd.DataFrame(reduced_embeddings,columns=["x","y"])
df["cluster"]=[str(c) for c in clusters]
to_plot = df.loc[df.cluster != "-1", :]

plt.scatter(to_plot.x,to_plot.y,c=to_plot.cluster.astype(int),alpha=0.6,cmap="tab20b")
plt.show()

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    verbose = True
).fit(texts,embeddings)

