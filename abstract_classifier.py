import os
import zipfile
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import plotly.express as px
from scipy.spatial.distance import cdist
import pickle
import argparse

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(['and', 'in', 'the', 'etc', 'like'])

# Command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Extract XMLs, perform topic modeling, classify, and query similarities.")
    parser.add_argument('--zip_folder', default='/Users/sraghava/downloaded_xmls', help='Folder containing ZIP files')
    parser.add_argument('--output_dir', default='./output', help='Output directory for results')
    return parser.parse_args()

# Extract XML files from ZIPs
def extract_zip(zip_path, extract_folder):
    try:
        print(f"Extracting {zip_path}...")
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        xml_files = [os.path.join(extract_folder, f) for f in os.listdir(extract_folder) if f.lower().endswith('.xml')]
        print(f"Extracted {len(xml_files)} XML files from {zip_path}")
        return xml_files
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return []

# Extract abstract from XML
def extract_abstract(xml_path):
    try:
        tree = ET.parse(xml_path)
        abstract = tree.find(".//AbstractNarration")
        return abstract.text.strip() if abstract is not None and abstract.text else None
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

# Preprocess text (remove stop words, optional lemmatization)
def preprocess_text(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Perform zero-shot classification
def classify_abstracts(abstracts, classifier, labels=["positive", "negative", "neutral"]):
    classifications = []
    for abstract in tqdm(abstracts, desc="Classifying Abstracts"):
        try:
            result = classifier(abstract, candidate_labels=labels, multi_label=False)
            classifications.append(result['labels'][0])  # Take top label
        except Exception as e:
            print(f"Error classifying abstract: {e}")
            classifications.append("unknown")
    return classifications

# Query similar documents
def query_similar_documents(query, texts, embeddings, embedding_model, top_n=5):
    query_embedding = embedding_model.encode([query])[0]
    distances = cdist([query_embedding], embeddings, metric='cosine')[0]
    top_indices = np.argsort(distances)[:top_n]
    results = [(texts[i], distances[i], i) for i in top_indices]
    return results

def main():
    args = parse_args()
    xml_folder = args.zip_folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract XMLs from ZIPs
    downloaded_zips = [f for f in os.listdir(xml_folder) if f.endswith('.zip')]
    all_xml_files = []
    for zip_file in tqdm(downloaded_zips, desc="Extracting ZIPs"):
        xml_files = extract_zip(
            os.path.join(xml_folder, zip_file),
            os.path.join(xml_folder, f"{zip_file}_xml")
        )
        all_xml_files.extend(xml_files)
    print(f"Total XML files extracted: {len(all_xml_files)}")

    # Step 2: Extract abstracts
    abstracts = []
    for xml_file in tqdm(all_xml_files, desc="Extracting Abstracts"):
        abstract = extract_abstract(xml_file)
        if abstract:
            abstracts.append(abstract)
    print(f"Extracted {len(abstracts)} abstracts")

    # Save abstracts to file
    abstracts_df = pd.DataFrame(abstracts, columns=['abstract'])
    abstracts_df.to_csv(os.path.join(output_dir, 'abstracts.csv'), index=False)

    # Step 3: Preprocess texts
    texts = [preprocess_text(doc) for doc in abstracts]

    # Step 4: Generate embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_file = os.path.join(output_dir, 'embeddings.pkl')
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print("Loaded cached embeddings")
    else:
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print("Saved embeddings to cache")

    # Step 5: Topic modeling with BERTopic
    umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom')
    vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.95)
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=50,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # Save topic information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, 'topic_info.csv'), index=False)

    # Print sample documents for a specific topic
    cluster = 0
    print(f"\nSample documents for Topic {cluster}:")
    for index in np.where(topics == cluster)[0][:3]:
        print(f"Document {index}: {texts[index][:300]}...\n")

    # Step 6: Visualize topics (interactive with Plotly)
    reduced_embeddings = UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    df['cluster'] = [str(c) for c in topics]
    df['abstract'] = abstracts  # For hover text
    to_plot = df.loc[df['cluster'] != '-1', :]
    fig = px.scatter(
        to_plot, x='x', y='y', color='cluster',
        hover_data=['abstract'], title='Topic Clusters',
        color_continuous_scale='tab20b'
    )
    fig.write_html(os.path.join(output_dir, 'topic_plot.html'))
    print("Saved interactive topic plot to topic_plot.html")

    # Step 7: Zero-shot classification
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli',device_map="auto")
    labels = ['computer science', 'biology', 'geoscience', 'physics', 'chemistry', 'engineering','education']  # Customize as needed
    classifications = classify_abstracts(abstracts, classifier, labels)
    abstracts_df['classification'] = classifications
    abstracts_df['topic'] = topics
    abstracts_df.to_csv(os.path.join(output_dir, 'abstracts_classified.csv'), index=False)
    print("Saved classified abstracts to abstracts_classified.csv")

    # Step 8: Similarity querying
    query = input("Enter a query to find similar documents (e.g., 'machine learning'): ")
    similar_docs = query_similar_documents(query, texts, embeddings, embedding_model, top_n=5)
    print("\nTop similar documents:")
    for doc, distance, idx in similar_docs:
        print(f"Document {idx} (Cosine Distance: {distance:.4f}):")
        print(f"{abstracts[idx][:300]}...\n")

if __name__ == '__main__':
    main()