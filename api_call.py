import requests
import xml.etree.ElementTree as ET
import json
import csv
from collections import defaultdict

# Fetch GEO dataset IDs linked to given PMIDs using the ELink API.
def get_geo_ids_from_pmids(pmid_list):
    pmid_to_geo = defaultdict(list)
    geo_to_pmid = defaultdict(list)
    
    for pmid in pmid_list:
        elink_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?"
            f"dbfrom=pubmed&db=gds&id={pmid}&linkname=pubmed_gds"
        )
        response = requests.get(elink_url)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for link in root.findall(".//LinkSetDb/Link/Id"):
                geo_id = link.text
                pmid_to_geo[pmid].append(geo_id)
                geo_to_pmid[geo_id].append(pmid)

    return pmid_to_geo, geo_to_pmid

# Fetch GEO dataset details for given GEO IDs using the ESummary API.
def get_geo_dataset_details(geo_ids):
    geo_ids_str = ",".join(geo_ids)
    esummary_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        f"db=gds&id={geo_ids_str}&retmode=json"
    )
    response = requests.get(esummary_url)

    if response.status_code == 200:
        return response.json().get("result", {})
    return {}

# Remove commas from text to prevent CSV format issues.
def sanitize(text):
    return text.replace(",", " ") if text else "N/A"

# Write mapping dictionary to CSV.
def write_mapping_file(mapping, filename, key_header, value_header):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key_header, value_header])
        for key, values in mapping.items():
            for value in values:
                writer.writerow([key, value])

def main():
    # Read PMIDs from file
    with open("PMIDs_list.txt") as f:
        pmid_list = [line.strip() for line in f if line.strip()]

    # Get mappings
    pmid_to_geo, geo_to_pmid = get_geo_ids_from_pmids(pmid_list)

    if not pmid_to_geo:
        print("No GEO IDs found.")
        return

    # Save mapping file
    write_mapping_file(geo_to_pmid, "GEO_to_PMID.csv", "GEO ID", "PMID")

    # Flatten all unique GEO IDs
    unique_geo_ids = list(geo_to_pmid.keys())

    # Get dataset details
    geo_data = get_geo_dataset_details(unique_geo_ids)

    # Extract relevant fields
    extracted_data = []
    for geo_id in unique_geo_ids:
        record = geo_data.get(geo_id, {})
        extracted_data.append({
            "GEO ID": geo_id,
            "Title": sanitize(record.get("title", "N/A")),
            "Experiment type": sanitize(record.get("gdstype", "N/A")),
            "Summary": sanitize(record.get("summary", "N/A")),
            "Organism": sanitize(record.get("taxon", "N/A"))
        })

    # Save GEO dataset details
    csv_file = "GEO_datasets_output.csv"
    with open(csv_file, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["GEO ID", "Title", "Experiment type", "Summary", "Organism"])
        writer.writeheader()
        writer.writerows(extracted_data)

if __name__ == "__main__":
    main()
