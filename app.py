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
            selected_file = DEFAULT_PMID_FILE
        else:
            pmids = request.form["pmids"]

            # Save input PMIDs to file
            with open(PMID_FILE, "w") as f:
                f.write(pmids.strip())
            selected_file = PMID_FILE

        return redirect(url_for("loading", pmid_file=os.path.basename(selected_file)))

    return render_template("index.html")

# Loading page shown while processing data
@app.route("/loading")
def loading():
    pmid_file = request.args.get("pmid_file")
    return render_template("loading.html", pmid_file=pmid_file)


# Backend processing route: calls API + model, then returns result
@app.route("/process/<pmid_file>")
def process(pmid_file):
    
    # Write file name to disk so api_call can access it
    api_call_path = os.path.join(".", pmid_file)

    # Run the API call script (it will read the file, fetch and save outputs)
    with open(api_call_path) as f:
        pmids = [line.strip() for line in f if line.strip()]
    if not pmids:
        return "No PMIDs provided.", 400

    # Trigger data fetch and save
    api_call.main()

    # Trigger model and get HTML figure
    graph_html = model.main()
    return render_template("result.html", graph_html=graph_html)

if __name__ == "__main__":
    app.run(debug=True)
