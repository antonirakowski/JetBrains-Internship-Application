import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from spacy.cli import download
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
import plotly.graph_objects as go
import random
import math

# Setting seeds
random.seed(42)
np.random.seed(42)

# Run this only once!
# nltk.download("stopwords")
# nltk.download("punkt")
# download("en_core_web_sm")

# Function for removing punctuation
def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Loading stop words
stop_words = set(stopwords.words("english"))

# Function for removing stop words
def remove_stopwords(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function for converting text to lower
def to_lower(text):
    return text.lower()

# Loading English NLP model
nlp = spacy.load("en_core_web_sm")

# Function for lemmatizing text
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# Function for finding the optimal number of clusters in
# KMeans algorithm trained on some data set X
def find_best_k(X, k_range=range(2, 15)):
    best_k = None
    best_score = -1
    best_model = None

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)

        # Only compute silhouette score if we have more than 1 cluster
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans

    return best_model

def main():

    # Loading the data
    df = pd.read_csv("GEO_datasets_output.csv")

    # Assigning the DataFrame length to n
    n = df.shape[0]

    # Connecting all text columns
    df["Connected text"] = df["Title"] + " " + df["Experiment type"] + " " + df["Summary"] + df["Organism"]
    df = df[["GEO ID", "Connected text"]]

    # Removing punctuation
    df["Connected text"] = df["Connected text"].apply(remove_punctuation)

    # Removing stop words
    df["Connected text"] = df["Connected text"].apply(remove_stopwords)

    # Applying lowercase to text
    df["Connected text"] = df["Connected text"].apply(to_lower)

    # Lemmatizing text
    df["Connected text"] = df["Connected text"].apply(lemmatize_text)

    # Applying TF-IDF
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3)) # Configuring for unigrams, bigrams, trigrams
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["Connected text"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Applying MinMaxScaler
    min_max_scaler = MinMaxScaler()
    tfidf_df = min_max_scaler.fit_transform(tfidf_df)

    # Applying PCA
    pca = PCA(n_components= math.ceil(2 * math.log(n))) # Setting the number of components to 2log(n)
    X_train = pca.fit_transform(tfidf_df)

    # Training KMeans with optimal (in the sense of Silhouette) number of clusters k
    kmeans = find_best_k(X_train)
    kmeans_labels = kmeans.fit_predict(X_train)

    # Creating DataFrame with columns PMID and Cluster for visualisation purposes
    clusters = pd.DataFrame(kmeans_labels)
    clusters.columns = ["Cluster"]
    clusters = pd.concat([df["GEO ID"], clusters], axis=1)
    geo_to_pmid = pd.read_csv("GEO_to_PMID.csv")
    merged = pd.merge(geo_to_pmid, clusters, on="GEO ID", how="inner").drop(["GEO ID"], axis = 1)

    # Create Graphb
    G = nx.Graph()

    # Add cluster nodes
    clusters = merged["Cluster"].unique()
    for cluster in clusters:
        G.add_node(f"Cluster {cluster}", type="cluster")

    # Add PMID nodes and edges to clusters
    for pmid, group in merged.groupby("PMID"):
        G.add_node(pmid, type="pmid")
        for cluster in group["Cluster"].unique():
            G.add_edge(pmid, f"Cluster {cluster}")

    # Node positions using spring layout
    pos = nx.spring_layout(G, k=0.4, seed=42)

    # Separate nodes by type
    node_x, node_y, node_text, node_color = [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        if G.nodes[node]["type"] == "cluster":
            node_color.append("orange")
        else:
            node_color.append("skyblue")

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Plotly figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=10,
            line=dict(width=2)
        )
    ))

    fig.update_layout(
        title="PMID to Cluster Network Graph",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=20, r=20, t=40, b=20),
        height=700
    )

    # Returning the visualisation
    return fig.to_html(full_html=False)

if __name__ == "__main__":
    main()