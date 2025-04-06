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

### ⚠️ **Important Notes:**

- `api_call.py` reads from a file called `PMIDs_list.txt`, which is automatically generated by `app.py` based on user input or default PMIDs.  
  **Make sure to run the Flask app (`app.py`) first to generate this file** before executing `api_call.py` directly.

- `api_call.py` fetches metadata and writes the results into CSV files.  
  These CSV files are required inputs for both:
  - `model.py`, which processes the data and creates interactive visualizations.
  - `model_creation.ipynb`, which contains the full modeling workflow and exploratory analysis.

> ✅ So always run `api_call.py` (directly or through the app) before using the model scripts.

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
