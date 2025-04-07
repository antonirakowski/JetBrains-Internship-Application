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

---

## ⚙️ Setup Instructions

### 🐍 Python Version
This project was built and tested with **Python 3.12.2**. Make sure you're using this version (or a compatible one) for best results.

### 📦 Package requirements

The specific tools, libraries, and frameworks used throughout the project are listed in the [`requirements.txt`](requirements.txt) file.

### 🔧 Installation Steps 

1. **Clone the repository**:
   ```bash
   git clone https://github.com/antonirakowski/JetBrains-Internship-Application;
   cd JetBrains-Internship-Application
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run set-up script (needed only once)**
   ```bash
   python setup.py
   ```

   The script includes:
   ```python
   import nltk
   from spacy.cli import download

   nltk.download("stopwords")
   nltk.download("punkt")
   nltk.download("punkt_tab")
   download("en_core_web_sm")
   ```
4. **Start the app**
   ```bash
   python app.py
   ```
5. **Now feel free to explore directly:** <br>
`model_creation.ipynb` <br>
`model.py`<br>
`api_call.py` 

### ⚠️ **Important Notes:**

- `api_call.py` reads from a file called `PMIDs_list.txt`, which is automatically generated by `app.py` based on user input or a default list of PMIDs.  

- `api_call.py` fetches metadata and saves it into CSV files.  
  These files are essential for:
  - `model.py`.
  - `model_creation.ipynb`.

Thus, **make sure to run the Flask app** (`app,py`) **before launching** 
`model_creation.ipynb`, `model.py`, `api_call.py` independently for the first time. After running the app once, you're free to explore the Python scripts and Jupyter notebook directly.