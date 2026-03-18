import os
import json
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    parser = argparse.ArgumentParser(description="Process NSF JSON award abstracts, perform topic modeling, classify, query similarities, and analyze Directorate relationships.")
    parser.add_argument('--json_folder', default='/Users/sraghava/json_dirs', help='Folder containing subdirectories with JSON files')
    parser.add_argument('--output_dir', default='./output', help='Output directory for results')
    parser.add_argument('--db', default=None, help='Path to SQLite DB from nsf_xml_parser.py (XML mode). If omitted, uses --json_folder.')
    return parser.parse_args()

# Collect JSON files from subdirectories
def collect_json_files(json_folder):
    try:
        print(f"Collecting JSON files from {json_folder}...")
        all_json_files = []
        subdirs = [os.path.join(json_folder, d) for d in os.listdir(json_folder) 
                   if os.path.isdir(os.path.join(json_folder, d))]
        for subdir in tqdm(subdirs, desc="Processing Subdirectories"):
            json_files = glob.glob(os.path.join(subdir, "*.json"))
            all_json_files.extend(json_files)
            print(f"Found {len(json_files)} JSON files in {subdir}")
        print(f"Total JSON files found: {len(all_json_files)}")
        return all_json_files
    except Exception as e:
        print(f"Error collecting JSON files from {json_folder}: {e}")
        return []

# Extract abstract and Directorate from JSON
def extract_abstract_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Adjust keys based on NSF JSON structure (e.g., "abstractText", "directorate")
        abstract = data.get('awd_abstract_narration')
        directorate = data.get('dir_abbr')  # Adjust if nested, e.g., data.get('award', {}).get('directorate')
        title = data.get('awd_titl_tx')
        if abstract and directorate:
            return abstract.strip(), directorate.strip(), title.strip()
        return None, None, None
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return None, None, None

# Preprocess text (remove stop words)
def preprocess_text(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Perform zero-shot classification
def classify_abstracts(abstracts, classifier, labels=["positive", "negative", "neutral"]):
    classifications = []
    for abstract in tqdm(abstracts, desc="Classifying Abstracts"):
        try:
            result = classifier(abstract, candidate_labels=labels, multi_label=False)
            classifications.append(result['labels'][0])
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

# Analyze topic distribution by Directorate
def analyze_topic_by_directorate(topics, directorates, topic_info, output_dir):
    df = pd.DataFrame({'topic': topics, 'directorate': directorates})
    topic_dist = df.groupby(['directorate', 'topic']).size().unstack(fill_value=0)
    
    # Normalize by Directorate to get proportions
    topic_dist_norm = topic_dist.div(topic_dist.sum(axis=1), axis=0)
    
    # Save to CSV
    topic_dist.to_csv(os.path.join(output_dir, 'topic_by_directorate.csv'))
    topic_dist_norm.to_csv(os.path.join(output_dir, 'topic_by_directorate_normalized.csv'))
    
    # Plot topic distribution
    plt.figure(figsize=(12, 8))
    topic_dist.plot(kind='bar', stacked=True, cmap='tab20')
    plt.title('Topic Distribution by Directorate')
    plt.xlabel('Directorate')
    plt.ylabel('Number of Awards')
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_by_directorate.png'))
    plt.close()
    
    return topic_dist, topic_dist_norm

# Compute and visualize Directorate similarities based on topics
def compute_directorate_similarities(topic_dist_norm, output_dir):
    # Compute cosine similarity between Directorates based on topic proportions
    similarity_matrix = 1 - cdist(topic_dist_norm.values, topic_dist_norm.values, metric='cosine')
    similarity_df = pd.DataFrame(similarity_matrix, index=topic_dist_norm.index, columns=topic_dist_norm.index)
    
    # Save to CSV
    similarity_df.to_csv(os.path.join(output_dir, 'directorate_similarity.csv'))
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Directorate Similarity Based on Topic Distributions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'directorate_similarity_heatmap.png'))
    plt.close()
    
    return similarity_df

def main():
    args = parse_args()
    json_folder = args.json_folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.db:
        # --- XML / SQLite mode ---
        from nsf_xml_parser import NSFAwardDB
        print(f"Loading records from DB: {args.db}")
        records = NSFAwardDB(args.db).get_pipeline_records(require_abstract=True)
        abstracts    = [r['abstract']    for r in records]
        directorates = [r['directorate'] for r in records]
        titles       = [r['title']       for r in records]
        # Enriched KG fields (available for downstream use)
        divisions    = [r['division']                for r in records]
        pe_codes     = [r['program_element_codes']   for r in records]
        pr_codes     = [r['program_reference_codes'] for r in records]
        print(f"Loaded {len(abstracts)} records from DB")
    else:
        # --- Legacy JSON mode ---
        # Step 1: Collect JSON files from subdirectories
        all_json_files = collect_json_files(json_folder)
        print(f"Total JSON files collected: {len(all_json_files)}")

        # Step 2: Extract abstracts and Directorates
        abstracts = []
        directorates = []
        titles = []
        for json_file in tqdm(all_json_files, desc="Extracting Abstracts and Directorates"):
            abstract, directorate, title = extract_abstract_from_json(json_file)
            if abstract and directorate and title:
                abstracts.append(abstract)
                directorates.append(directorate)
                titles.append(title)
        print(f"Extracted {len(abstracts)} abstracts with Directorates")
        divisions = pe_codes = pr_codes = [None] * len(abstracts)

    # Save abstracts and Directorates to file
    abstracts_df = pd.DataFrame({'abstract': abstracts, 'directorate': directorates})
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

    # Step 6: Analyze topic distribution by Directorate
    topic_dist, topic_dist_norm = analyze_topic_by_directorate(topics, directorates, topic_info, output_dir)
    print("Saved topic distribution by Directorate to topic_by_directorate.csv and topic_by_directorate_normalized.csv")

    # Step 7: Compute Directorate similarities
    similarity_df = compute_directorate_similarities(topic_dist_norm, output_dir)
    print("Saved Directorate similarity matrix to directorate_similarity.csv")

    # Step 8: Visualize topics (interactive with Plotly)
    reduced_embeddings = UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    df['cluster'] = [str(c) for c in topics]
    df['abstract'] = abstracts
    df['directorate'] = directorates
    to_plot = df.loc[df['cluster'] != '-1', :]
    fig = px.scatter(
        to_plot, x='x', y='y', color='cluster', symbol='directorate',
        hover_data=['abstract', 'directorate'], title='Topic Clusters by Directorate',
        color_continuous_scale='tab20b'
    )
    fig.write_to_html(os.path.join(output_dir, 'topic_plot.html'))
    print("Saved interactive topic plot to topic_plot.html")

    # Step 9: Zero-shot classification
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    labels = ['positive', 'negative', 'neutral']
    classifications = classify_abstracts(abstracts, classifier, labels)
    abstracts_df['classification'] = classifications
    abstracts_df['topic'] = topics
    abstracts_df.to_csv(os.path.join(output_dir, 'abstracts_classified.csv'), index=False)
    print("Saved classified abstracts to abstracts_classified.csv")

    # Step 10: Similarity querying
    query = input("Enter a query to find similar documents (e.g., 'machine learning'): ")
    similar_docs = query_similar_documents(query, texts, embeddings, embedding_model, top_n=5)
    print("\nTop similar documents:")
    for doc, distance, idx in similar_docs:
        print(f"Document {idx} (Cosine Distance: {distance:.4f}, Directorate: {directorates[idx]}):")
        print(f"{abstracts[idx][:300]}...\n")

if __name__ == '__main__':
    main()