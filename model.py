import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from spacy.cli import download
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        # Only compute silhouette score if we have more than 1 cluster
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans

    return best_model

# Function used to create NetworkX/Plotly visualisations of the clusters
def create_cluster_graph(
    df,
    node_column,
    mapping_df,
    mapping_key,
    mapping_value,
    node_label_prefix,
    node_color,
    mapping_hover_label,
    title
):
    # Step 1: Create the graph
    G = nx.Graph()

    # Step 2: Add cluster nodes
    clusters = df["Cluster"].unique()
    for cluster in clusters:
        G.add_node(f"Cluster {cluster}", type="cluster")

    # Step 3: Add primary nodes (GEO ID or PMID) and connect to clusters
    for primary_node, group in df.groupby(node_column):
        G.add_node(primary_node, type="primary")
        for cluster in group["Cluster"].unique():
            G.add_edge(primary_node, f"Cluster {cluster}")

    # Step 4: Mapping primary node to secondary IDs
    node_to_secondary = mapping_df.groupby(mapping_key)[mapping_value].apply(list).to_dict()

    # Step 5: Node positions
    pos = nx.spring_layout(G, k=0.4, seed=42)

    # Step 6: Node data
    node_x, node_y, node_text, node_hovertext, node_color_list = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if G.nodes[node]["type"] == "cluster":
            node_text.append(str(node))
            node_hovertext.append(str(node))
            node_color_list.append("orange")
        else:
            node_text.append(str(node))
            secondary_items = node_to_secondary.get(node, [])
            secondary_str = ", ".join(str(s) for s in secondary_items) if secondary_items else "None"
            hover_text = f"{node_label_prefix}: {node}<br>{mapping_hover_label}: {secondary_str}"
            node_hovertext.append(hover_text)
            node_color_list.append(node_color)

    # Step 7: Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Step 8: Plotly figure
    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        hovertext=node_hovertext,
        hoverinfo="text",
        textposition="top center",
        marker=dict(
            color=node_color_list,
            size=10,
            line=dict(width=2)
        )
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=20, r=20, t=40, b=20),
        height=700,
        width = 1200
    )

    return fig.to_html(full_html=False)

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

    # Scaling
    scaler = StandardScaler()
    tfidf_df = scaler.fit_transform(tfidf_df)

    # Applying PCA
    pca = PCA(n_components= math.ceil(math.log(n))) # Setting the number of components to 2log(n)
    X_train = pca.fit_transform(tfidf_df)

    # Training KMeans with optimal (in the sense of Silhouette) number of clusters k
    kmeans = find_best_k(X_train)
    kmeans_labels = kmeans.fit_predict(X_train)

    # Creating appropriate DataFrames for visualisation purposes
    clusters = pd.DataFrame(kmeans_labels)
    clusters.columns = ["Cluster"]
    geo_clusters = pd.concat([df["GEO ID"], clusters], axis=1)
    geo_to_pmid = pd.read_csv("GEO_to_PMID.csv")
    merged = pd.merge(geo_to_pmid, geo_clusters, on="GEO ID", how="inner").drop(["GEO ID"], axis = 1)

    # Generating a graph where GEO IDs and clusters are nodes. A GEO ID node is connected to a cluster node 
    # if and only if the GEO ID belongs to that cluster. Hovering over a GEO ID node will display the associated PMID(s).
    geo_html = create_cluster_graph(
    df=geo_clusters,
    node_column="GEO ID",
    mapping_df=geo_to_pmid,
    mapping_key="GEO ID",
    mapping_value="PMID",
    node_label_prefix="GEO ID",
    node_color="lightgreen",
    mapping_hover_label="PMID(s)",
    title="GEO ID to Cluster Network Graph with PMIDs on Hover"
    )

    # Generating an analogous graph where PMIDs and clusters are nodes.
    # Each PMID is connected to its cluster(s), and hovering over a PMID node will display the related GEO ID(s).
    pmid_html = create_cluster_graph(
        df=merged,
        node_column="PMID",
        mapping_df=geo_to_pmid,
        mapping_key="PMID",
        mapping_value="GEO ID",
        node_label_prefix="PMID",
        node_color="skyblue",
        mapping_hover_label="GEO ID(s)",
        title="PMID to Cluster Network Graph with GEO IDs on Hover"
    )

    # Returning the visualisation
    return geo_html, pmid_html

if __name__ == "__main__":
    main()