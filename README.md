# 🚀 JetBrains Internship Application

## 🧩 Project Overview

This project is a web-based application that integrates biomedical data retrieval and clustering analysis. The core goals are:

1. **Data Fetching**: Retrieve GEO datasets related to user-provided PMIDs using NCBI's ELink and ESummary APIs.
2. **Clustering Model**: Group datasets based on textual features (title, experiment type, summary, and organism) using TF-IDF and clustering algorithms.
3. **User Interface**: Provide a clean, interactive, and user-friendly experience using a Flask web interface.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `model_creation.ipynb` | Jupyter notebook showcasing the complete modeling process, including experimentation, rationale, and results. |
| `model.py` | Script that runs the final model, clusters the data, and visualizes results using Plotly and NetworkX. |
| `api_call.py` | Script that fetches GEO dataset metadata (GEO IDs, titles, summaries, organisms, experiment types) based on provided PMIDs using NCBI APIs. |
| `app.py` | Flask application allowing users to input PMIDs or use a default list, fetch data, run the model, and display interactive visualizations. |

The specific tools, libraries, and frameworks used throughout the project are listed in the [`requirements.txt`](requirements.txt) file.

---

## ⚙️ Setup Instructions

Before running the project for the first time, make sure to download the required NLTK and spaCy resources:

```python
import nltk
from spacy.cli import download

nltk.download("stopwords")
nltk.download("punkt")
download("en_core_web_sm")
