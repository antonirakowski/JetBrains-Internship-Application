from flask import Flask, render_template, request, redirect, url_for
import api_call
import model
import os

app = Flask(__name__)

# File names for storing PMIDs
PMID_FILE = "PMIDs_list.txt"
DEFAULT_PMID_FILE = "PMIDs_default.txt"

# Main page: handles form for user-submitted PMIDs or default use
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "use_default" in request.form:
            # Copy contents of DEFAULT_PMID_FILE into PMID_FILE
            with open(DEFAULT_PMID_FILE, "r") as default_f:
                default_pmids = default_f.read().strip()
            with open(PMID_FILE, "w") as f:
                f.write(default_pmids)
        else:
            # Write user input into PMID_FILE
            pmids = request.form["pmids"]
            with open(PMID_FILE, "w") as f:
                f.write(pmids.strip())

        return redirect(url_for("loading", pmid_file=os.path.basename(PMID_FILE)))

    return render_template("index.html")

# Loading page shown while processing data
@app.route("/loading")
def loading():
    pmid_file = request.args.get("pmid_file")
    return render_template("loading.html", pmid_file=pmid_file)

# Backend processing route: calls API + model, then returns result
@app.route("/process/<pmid_file>")
def process(pmid_file):
    api_call_path = os.path.join(".", pmid_file)

    # Read PMIDs from file
    with open(api_call_path) as f:
        pmids = [line.strip() for line in f if line.strip()]
    if not pmids:
        return "No PMIDs provided.", 400

    # Trigger data fetch and model processing
    api_call.main()
    geo_html, pmid_html = model.main()

    return render_template("result.html", geo_html=geo_html, pmid_html=pmid_html)

if __name__ == "__main__":
    app.run(debug=True)