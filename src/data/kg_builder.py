"""Build PrimeKG-compatible knowledge graphs from source databases.

Supports selective database inclusion for partial builds when some
databases require registration. Adapted from the PrimeKG repository
(https://github.com/mims-harvard/PrimeKG) with parameterized paths
and automated assembly.

Usage:
    from src.data.kg_builder import download_sources, build_kg
    download_sources(config, output_dir)
    kg = build_kg(data_dir, config)
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EDGE_COLUMNS = [
    "relation", "display_relation",
    "x_id", "x_type", "x_name", "x_source",
    "y_id", "y_type", "y_name", "y_source",
]


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, gunzip: bool = False) -> Path:
    """Download a file from URL. Gunzip if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"  Already exists: {dest}")
        return dest

    gz_dest = Path(str(dest) + ".gz") if gunzip else dest
    logger.info(f"  Downloading {url}")

    # Use a proper request with User-Agent to avoid 403 errors
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (MCGL-Project/1.0)")
    with urllib.request.urlopen(req, timeout=120) as resp, open(str(gz_dest), "wb") as f_out:
        shutil.copyfileobj(resp, f_out)

    if gunzip:
        logger.info(f"  Decompressing {gz_dest}")
        with gzip.open(str(gz_dest), "rb") as f_in, open(str(dest), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_dest.unlink()

    logger.info(f"  Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_gene_names(output_dir: Path) -> None:
    """Download HGNC gene name vocabulary."""
    url = (
        "https://www.genenames.org/cgi-bin/download/custom?"
        "col=gd_app_sym&col=gd_app_name&col=gd_pub_acc_ids&col=gd_pub_refseq_ids"
        "&col=gd_pub_eg_id&col=md_eg_id&col=md_prot_id&col=md_mim_id"
        "&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"
    )
    dest = output_dir / "vocab" / "gene_names.csv"
    _download(url, dest)

    # Also download gene_map (NCBI->UniProt)
    url2 = (
        "https://www.genenames.org/cgi-bin/download/custom?"
        "col=md_eg_id&col=md_prot_id"
        "&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"
    )
    _download(url2, output_dir / "vocab" / "gene_map.csv")


def download_bgee(output_dir: Path) -> None:
    """Download Bgee gene expression data."""
    url = "https://www.bgee.org/ftp/current/download/calls/expr_calls/Homo_sapiens_expr_advanced.tsv.gz"
    _download(url, output_dir / "bgee" / "Homo_sapiens_expr_advanced.tsv", gunzip=True)


def download_ctd(output_dir: Path) -> None:
    """Download CTD exposure events."""
    url = "https://ctdbase.org/reports/CTD_exposure_events.csv.gz"
    _download(url, output_dir / "ctd" / "CTD_exposure_events.csv", gunzip=True)


def download_disgenet(output_dir: Path, api_key: str | None = None) -> None:
    """Download DisGeNET curated gene-disease associations."""
    dest = output_dir / "disgenet" / "curated_gene_disease_associations.tsv"
    if dest.exists():
        logger.info(f"  Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try API download with key
    if api_key:
        logger.info("  Downloading DisGeNET via API with key")
        url = "https://www.disgenet.org/api/gda/source/CURATED"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "text/tab-separated-values")
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read().decode("utf-8")
            with open(str(dest), "w") as f:
                f.write(data)
            logger.info(f"  Saved to {dest}")
            return
        except Exception as e:
            logger.warning(f"  API download failed: {e}. Trying static URL...")

    # Fallback: static URL (may require auth for newer versions)
    url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz"
    _download(url, dest, gunzip=True)


def download_go(output_dir: Path) -> None:
    """Download Gene Ontology OBO file."""
    url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    _download(url, output_dir / "go" / "go-basic.obo")


def download_gene2go(output_dir: Path) -> None:
    """Download NCBI gene2go annotations."""
    url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
    _download(url, output_dir / "ncbigene" / "gene2go", gunzip=True)


def download_hpo(output_dir: Path) -> None:
    """Download Human Phenotype Ontology OBO file."""
    url = "http://purl.obolibrary.org/obo/hp.obo"
    _download(url, output_dir / "hpo" / "hp.obo")


def download_hpoa(output_dir: Path) -> None:
    """Download HPO disease-phenotype annotations."""
    url = "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa"
    _download(url, output_dir / "hpo" / "phenotype.hpoa")


def download_mondo(output_dir: Path) -> None:
    """Download MONDO disease ontology OBO file."""
    url = "http://purl.obolibrary.org/obo/mondo.obo"
    _download(url, output_dir / "mondo" / "mondo.obo")


def download_reactome(output_dir: Path) -> None:
    """Download Reactome pathway data."""
    base = "https://reactome.org/download/current"
    d = output_dir / "reactome"
    _download(f"{base}/ReactomePathways.txt", d / "ReactomePathways.txt")
    _download(f"{base}/ReactomePathwaysRelation.txt", d / "ReactomePathwaysRelation.txt")
    _download(f"{base}/NCBI2Reactome.txt", d / "NCBI2Reactome.txt")


def download_uberon(output_dir: Path) -> None:
    """Download UBERON anatomy ontology OBO file."""
    url = "http://purl.obolibrary.org/obo/uberon/ext.obo"
    _download(url, output_dir / "uberon" / "ext.obo")


DOWNLOAD_FNS = {
    "gene_names": download_gene_names,
    "bgee": download_bgee,
    "ctd": download_ctd,
    "disgenet": download_disgenet,
    "gene_ontology": download_go,
    "gene2go": download_gene2go,
    "hpo": download_hpo,
    "hpoa": download_hpoa,
    "mondo": download_mondo,
    "reactome": download_reactome,
    "uberon": download_uberon,
}


def download_sources(config: dict, output_dir: str | Path) -> None:
    """Download all enabled databases from config."""
    output_dir = Path(output_dir)
    databases = config.get("databases", {})

    # Always download gene_names first (needed by many processors)
    if databases.get("gene_names", {}).get("enabled", True):
        logger.info("Downloading gene_names vocabulary...")
        download_gene_names(output_dir)

    for name, db_conf in databases.items():
        if name == "gene_names" or not db_conf.get("enabled", False):
            continue
        if name not in DOWNLOAD_FNS:
            logger.info(f"Skipping {name} (no download function)")
            continue

        logger.info(f"Downloading {name}...")
        fn = DOWNLOAD_FNS[name]

        try:
            if name == "disgenet":
                api_key = os.environ.get("DISGENET_API_KEY", db_conf.get("api_key"))
                fn(output_dir, api_key=api_key)
            else:
                fn(output_dir)
        except Exception as e:
            logger.error(f"  FAILED to download {name}: {e}")
            logger.error(f"  You can retry later with --skip-download after manually fixing.")

    logger.info("All downloads complete.")


# ---------------------------------------------------------------------------
# Processing functions (adapted from PrimeKG processing_scripts/)
# ---------------------------------------------------------------------------

def load_gene_names(data_dir: Path) -> pd.DataFrame:
    """Load HGNC gene name vocabulary."""
    path = data_dir / "vocab" / "gene_names.csv"
    df = pd.read_csv(str(path), sep="\t", low_memory=False)
    # Standardize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "approved symbol" in cl or c == "Approved symbol":
            col_map[c] = "symbol"
        elif "ncbi gene id" in cl and "supplied" not in cl:
            col_map[c] = "ncbi_id"
    df = df.rename(columns=col_map)
    if "symbol" not in df.columns or "ncbi_id" not in df.columns:
        # Try positional
        df.columns = ["symbol", "name", "accession", "refseq", "ncbi_id",
                       "ncbi_id_supplied", "uniprot_id", "omim_id"][:len(df.columns)]
    df = df.dropna(subset=["ncbi_id"]).copy()
    df["ncbi_id"] = df["ncbi_id"].astype(int).astype(str)
    return df[["symbol", "ncbi_id"]].drop_duplicates()


def process_bgee(data_dir: Path) -> pd.DataFrame:
    """Process Bgee gene expression data. Adapted from PrimeKG bgee.py."""
    path = data_dir / "bgee" / "Homo_sapiens_expr_advanced.tsv"
    df = pd.read_csv(str(path), sep="\t", low_memory=False)
    df = df[df["Anatomical entity ID"].str.startswith("UBERON")]
    df = df[["Gene ID", "Gene name", "Anatomical entity ID",
             "Anatomical entity name", "Expression", "Call quality", "Expression rank"]]
    df = df.rename(columns={
        "Gene ID": "gene_id", "Gene name": "gene_name",
        "Anatomical entity ID": "anatomy_id", "Anatomical entity name": "anatomy_name",
        "Expression": "expression", "Call quality": "call_quality",
        "Expression rank": "expression_rank",
    })
    df = df.query('call_quality=="gold quality"')
    df = df.query("expression_rank<25000")
    df = df[~df["anatomy_id"].str.contains("\u2229")]  # Remove cell-type-in-tissue rows
    df["anatomy_id"] = [str(int(x.split(":")[1])) for x in df["anatomy_id"].values]
    return df


def process_ctd(data_dir: Path) -> pd.DataFrame:
    """Process CTD exposure data. Adapted from PrimeKG ctd.py."""
    path = data_dir / "ctd" / "CTD_exposure_events.csv"
    with open(str(path), "r") as f:
        lines = f.readlines()

    # Find the fields header line
    field_next = False
    fields_line = None
    for line in lines:
        if line.startswith("# Fields"):
            field_next = True
            continue
        if field_next:
            fields_line = line
            break

    if fields_line is None:
        # Try reading directly (some versions have no comment header)
        return pd.read_csv(str(path), low_memory=False)

    cols = fields_line[2:-2].split(",") if fields_line.startswith("#") else fields_line.strip().split(",")
    # Write clean CSV
    clean_path = data_dir / "ctd" / "exposure_data.csv"
    with open(str(clean_path), "w") as f:
        f.write(",".join(cols) + "\n")
        for line in lines:
            if not line.startswith("#"):
                f.write(line + "\n")
    return pd.read_csv(str(clean_path), low_memory=False)


def process_mondo(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process MONDO OBO file. Returns (terms, parents, xrefs).

    Adapted from PrimeKG mondo.py but uses our own OBO parser
    to avoid goatools dependency for MONDO-specific parsing.
    """
    path = data_dir / "mondo" / "mondo.obo"
    terms, parents, xrefs = _parse_obo_ontology(str(path), prefix="MONDO:")
    terms.to_csv(str(data_dir / "mondo" / "mondo_terms.csv"), index=False)
    parents.to_csv(str(data_dir / "mondo" / "mondo_parents.csv"), index=False)
    xrefs.to_csv(str(data_dir / "mondo" / "mondo_references.csv"), index=False)
    return terms, parents, xrefs


def process_hpo(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process HPO OBO file. Returns (terms, parents, xrefs).

    Adapted from PrimeKG hpo.py.
    """
    path = data_dir / "hpo" / "hp.obo"
    terms, parents, xrefs = _parse_obo_ontology(str(path), prefix="HP:")
    terms.to_csv(str(data_dir / "hpo" / "hp_terms.csv"), index=False)
    parents.to_csv(str(data_dir / "hpo" / "hp_parents.csv"), index=False)
    xrefs.to_csv(str(data_dir / "hpo" / "hp_references.csv"), index=False)
    return terms, parents, xrefs


def _parse_obo_ontology(obo_path: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse an OBO ontology file into terms, parent-child, and cross-references.

    Works for MONDO, HPO, and similar OBO-format ontologies.
    Replaces the goatools-based mondo_obo_parser.py and hpo_obo_parser.py.
    """
    prefix_len = len(prefix)

    with open(obo_path, "r") as f:
        content = f.read()

    # Split into [Term] blocks
    blocks = content.split("[Term]\n")

    terms_list = []
    parents_list = []
    xrefs_list = []

    for block in blocks[1:]:  # skip header
        lines = block.split("\n")
        term_id = None
        name = None
        is_obsolete = False
        replaced_by = None
        term_parents = []
        term_xrefs = []

        for line in lines:
            if line.startswith("["):  # next section
                break
            if line.startswith("id: "):
                raw_id = line[4:].strip()
                if raw_id.startswith(prefix):
                    term_id = raw_id[prefix_len:]
                else:
                    # Skip terms that don't belong to this ontology
                    term_id = None
                    break
            elif line.startswith("name: "):
                name = line[6:].strip()
            elif line.startswith("is_a: "):
                parent_raw = line[6:].split("!")[0].strip()
                if parent_raw.startswith(prefix):
                    parents_list.append({"parent": parent_raw[prefix_len:], "child": term_id})
                # Skip cross-ontology parent references (e.g., BFO, GO in MONDO)
            elif line.startswith("xref: ") and not line.startswith("xref: url:"):
                xref_val = line[6:].split()[0].strip()
                if ":" in xref_val:
                    ont, ont_id = xref_val.split(":", 1)
                    xrefs_list.append({"ontology_id": ont_id, "ontology": ont, f"{prefix[:-1].lower()}_id": term_id})
            elif "closeMatch" in line:
                # Handle MONDO closeMatch references (UMLS, SNOMEDCT, MESH, etc.)
                ref = _parse_close_match(line)
                if ref and ":" in ref:
                    ont, ont_id = ref.split(":", 1)
                    xrefs_list.append({"ontology_id": ont_id, "ontology": ont, f"{prefix[:-1].lower()}_id": term_id})
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True
            elif line.startswith("replaced_by: "):
                rb = line[13:].strip()
                replaced_by = rb[prefix_len:] if rb.startswith(prefix) else rb

        if term_id is not None:
            terms_list.append({
                "id": term_id,
                "name": name,
                "is_obsolete": is_obsolete,
                "replacement_id": replaced_by,
            })

    terms_df = pd.DataFrame(terms_list)
    parents_df = pd.DataFrame(parents_list).drop_duplicates() if parents_list else pd.DataFrame(columns=["parent", "child"])

    # Determine the xref ID column name
    id_col = f"{prefix[:-1].lower()}_id"
    if xrefs_list:
        xrefs_df = pd.DataFrame(xrefs_list).drop_duplicates()
    else:
        xrefs_df = pd.DataFrame(columns=["ontology_id", "ontology", id_col])

    logger.info(f"  Parsed {obo_path}: {len(terms_df)} terms, {len(parents_df)} parent-child, {len(xrefs_df)} xrefs")
    return terms_df, parents_df, xrefs_df


def _parse_close_match(line: str) -> str | None:
    """Parse closeMatch lines from MONDO OBO into ontology:id format."""
    if "umls" in line:
        return "UMLS:" + line.split("/")[-1].strip()
    elif "snomedct" in line:
        return "SCTID:" + line.split("/")[-1].strip()
    elif "mesh" in line:
        return "MESH:" + line.split("/")[-1].strip()
    elif "medgen" in line:
        return "MEDGEN:" + line.split("/")[-1].strip()
    elif "meddra" in line:
        return "MedDRA:" + line.split("/")[-1].strip()
    elif "omim" in line:
        return "OMIM:" + line.split("/")[-1].strip()
    elif "DOID" in line:
        return "DOID:" + line.split(":")[-1].strip()
    elif "NCIT" in line:
        return "NCIT:" + line.split(":")[-1].strip()
    elif "Orphanet" in line:
        return "Orphanet:" + line.split(":")[-1].strip()
    return None


def process_hpoa(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process HPO annotations. Returns (positive, negative) disease-phenotype.

    Adapted from PrimeKG hpoa.py.
    """
    path = data_dir / "hpo" / "phenotype.hpoa"
    with open(str(path), "r") as f:
        lines = f.readlines()

    # Skip comment lines
    data_lines = [l for l in lines if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(data_lines)), sep="\t")

    records = []
    for _, row in df.iterrows():
        db_id = str(row.get("database_id", row.get("DatabaseID", "")))
        hpo_id = str(row.get("hpo_id", row.get("HPO_ID", "")))
        qualifier = str(row.get("qualifier", row.get("Qualifier", "")))

        if ":" not in db_id or ":" not in hpo_id:
            continue
        ont, ont_id = db_id.split(":", 1)
        hp_num = str(int(hpo_id.split(":")[1]))
        records.append({
            "hp_id": hp_num,
            "disease_ontology": ont,
            "disease_ontology_id": ont_id,
            "qualifier": qualifier,
        })

    df_all = pd.DataFrame(records)
    df_neg = df_all.query('qualifier=="NOT"')[["hp_id", "disease_ontology", "disease_ontology_id"]].drop_duplicates()
    df_pos = df_all.query('qualifier!="NOT"')[["hp_id", "disease_ontology", "disease_ontology_id"]].drop_duplicates()

    df_pos.to_csv(str(data_dir / "hpo" / "disease_phenotype_pos.csv"), index=False)
    df_neg.to_csv(str(data_dir / "hpo" / "disease_phenotype_neg.csv"), index=False)
    return df_pos, df_neg


def process_go(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process Gene Ontology OBO. Returns (terms, edges).

    Uses goatools for proper GO DAG parsing.
    """
    from goatools.obo_parser import GODag

    obo_path = str(data_dir / "go" / "go-basic.obo")
    godag = GODag(obo_path, prt=None)

    all_terms = set()
    for k, v in godag.items():
        all_terms.add((v.item_id, v.name, v.namespace))
    terms = pd.DataFrame(all_terms, columns=["go_term_id", "go_term_name", "go_term_type"])

    edges = set()
    for _, row in terms.iterrows():
        parent_id = row["go_term_id"]
        if parent_id in godag:
            for child in godag[parent_id].children:
                edges.add((parent_id, child.id))
    edges_df = pd.DataFrame(edges, columns=["x", "y"])

    # Convert GO IDs to integer strings (GO:0000001 -> "1")
    terms["go_term_id"] = [str(int(x.split(":")[1])) for x in terms["go_term_id"].values]
    edges_df["x"] = [str(int(x.split(":")[1])) for x in edges_df["x"].values]
    edges_df["y"] = [str(int(x.split(":")[1])) for x in edges_df["y"].values]

    terms.to_csv(str(data_dir / "go" / "go_terms_info.csv"), index=False)
    edges_df.to_csv(str(data_dir / "go" / "go_terms_relations.csv"), index=False)
    return terms, edges_df


def process_gene2go(data_dir: Path) -> pd.DataFrame:
    """Process NCBI gene2go annotations. Returns gene-GO associations.

    Uses goatools Gene2GoReader for proper parsing.
    """
    from goatools.anno.genetogo_reader import Gene2GoReader

    path = str(data_dir / "ncbigene" / "gene2go")
    reader = Gene2GoReader(filename=path, taxids=[9606])
    ns2assc = reader.get_ns2assc()

    associations = []
    ns_map = {"MF": "molecular_function", "BP": "biological_process", "CC": "cellular_component"}
    for ns_abbr, ns_full in ns_map.items():
        if ns_abbr in ns2assc:
            for gene, goterms in ns2assc[ns_abbr].items():
                for go in goterms:
                    associations.append((gene, go, ns_full))

    df = pd.DataFrame(associations, columns=["ncbi_gene_id", "go_term_id", "go_term_type"])
    df["go_term_id"] = [str(int(x.split(":")[1])) for x in df["go_term_id"].values]
    df.to_csv(str(data_dir / "ncbigene" / "protein_go_associations.csv"), index=False)
    return df


def process_reactome(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process Reactome data. Returns (terms, relations, ncbi_mapping).

    Adapted from PrimeKG reactome.py.
    """
    d = data_dir / "reactome"
    df_ncbi = pd.read_csv(str(d / "NCBI2Reactome.txt"), sep="\t",
                          names=["ncbi_id", "reactome_id", "url", "reactome_name", "evidence_code", "species"])
    df_ncbi = df_ncbi.query('species=="Homo sapiens"').drop(["url", "evidence_code", "species"], axis=1)
    df_ncbi = df_ncbi.drop_duplicates().reset_index(drop=True)
    df_ncbi.to_csv(str(d / "reactome_ncbi.csv"), index=False)

    df_terms = pd.read_csv(str(d / "ReactomePathways.txt"), sep="\t",
                           names=["reactome_id", "reactome_name", "species"])
    df_terms = df_terms.query('species=="Homo sapiens"').reset_index(drop=True)
    df_terms.to_csv(str(d / "reactome_terms.csv"), index=False)

    valid_ids = set(df_terms["reactome_id"].values)
    df_rels = pd.read_csv(str(d / "ReactomePathwaysRelation.txt"), sep="\t",
                          names=["reactome_id_1", "reactome_id_2"])
    df_rels = df_rels[df_rels["reactome_id_1"].isin(valid_ids) & df_rels["reactome_id_2"].isin(valid_ids)]
    df_rels = df_rels.reset_index(drop=True)
    df_rels.to_csv(str(d / "reactome_relations.csv"), index=False)

    return df_terms, df_rels, df_ncbi


def process_uberon(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process UBERON anatomy ontology. Returns (terms, rels, is_a).

    Adapted from PrimeKG uberon.py.
    """
    path = data_dir / "uberon" / "ext.obo"
    with open(str(path), "r") as f:
        content = f.read()

    blocks = content.split("[Term]\n")[1:]
    records = []
    for block in blocks:
        lines = block.split("\n")
        rec = {}
        for line in lines:
            if line.startswith("["):
                break
            if ": " in line:
                key, val = line.split(": ", 1)
                val = val.split(" !")[0]
                rec[key] = val
        if rec:
            records.append(rec)

    data = pd.DataFrame(records)

    # Parse relationship column
    relations = []
    for val in data.get("relationship", pd.Series(dtype=str)).values:
        if isinstance(val, str):
            parts = val.split(" ")[:2]
            relations.append(tuple(parts))
        else:
            relations.append((None, None))
    rel_df = pd.DataFrame(relations, columns=["relation_type", "relation_id"])
    df = pd.concat([data.reset_index(drop=True), rel_df.reset_index(drop=True)], axis=1)

    # Filter
    df = df[df.get("is_obsolete", pd.Series(dtype=str)) != "true"]
    df = df.dropna(subset=["is_a"])
    df = df[df["id"].str.startswith("UBERON")]
    df = df.reset_index(drop=True)

    def uberon_to_int(series):
        return [str(int(x.split(":")[1])) for x in series.values]

    df_terms = df[["id", "name"]].copy()
    df_terms.loc[:, "id"] = uberon_to_int(df_terms["id"])

    df_is_a = df[["id", "is_a"]].copy()
    df_is_a = df_is_a[df_is_a["is_a"].str.startswith("UBERON")]
    df_is_a.loc[:, "id"] = uberon_to_int(df_is_a["id"])
    df_is_a.loc[:, "is_a"] = [str(int(x.split(" {")[0].split(":")[1])) for x in df_is_a["is_a"].values]

    df_rels = df[["id", "relation_type", "relation_id"]].dropna()
    df_rels = df_rels[df_rels["relation_id"].str.startswith("UBERON")]
    df_rels.loc[:, "id"] = uberon_to_int(df_rels["id"])
    df_rels.loc[:, "relation_id"] = uberon_to_int(df_rels["relation_id"])

    df_terms.to_csv(str(data_dir / "uberon" / "uberon_terms.csv"), index=False)
    df_rels.to_csv(str(data_dir / "uberon" / "uberon_rels.csv"), index=False)
    df_is_a.to_csv(str(data_dir / "uberon" / "uberon_is_a.csv"), index=False)

    return df_terms, df_rels, df_is_a


# ---------------------------------------------------------------------------
# Edge assembly functions (adapted from build_graph.ipynb)
# ---------------------------------------------------------------------------

def clean_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize edge DataFrame: select columns, drop nulls/dupes/self-loops."""
    df = df[EDGE_COLUMNS].copy()
    df = df.dropna().drop_duplicates()
    # Remove self-loops
    mask = ~((df["x_id"] == df["y_id"]) & (df["x_type"] == df["y_type"]) &
             (df["x_source"] == df["y_source"]) & (df["x_name"] == df["y_name"]))
    return df[mask].reset_index(drop=True)


def build_edge_disease_disease(mondo_terms: pd.DataFrame, mondo_parents: pd.DataFrame) -> pd.DataFrame:
    """Build disease-disease edges from MONDO hierarchy."""
    df = pd.merge(mondo_parents, mondo_terms, "left", left_on="parent", right_on="id")
    df = df.rename(columns={"parent": "x_id", "name": "x_name"})
    df = pd.merge(df, mondo_terms, "left", left_on="child", right_on="id")
    df = df.rename(columns={"child": "y_id", "name": "y_name"})
    df["x_type"] = "disease"
    df["x_source"] = "MONDO"
    df["y_type"] = "disease"
    df["y_source"] = "MONDO"
    df["relation"] = "disease_disease"
    df["display_relation"] = "parent-child"
    return clean_edges(df)


def build_edge_phenotype_phenotype(hp_terms: pd.DataFrame, hp_parents: pd.DataFrame) -> pd.DataFrame:
    """Build phenotype-phenotype edges from HPO hierarchy."""
    df = pd.merge(hp_parents, hp_terms, "left", left_on="parent", right_on="id")
    df = df.rename(columns={"parent": "x_id", "name": "x_name"})
    df = pd.merge(df, hp_terms, "left", left_on="child", right_on="id")
    df = df.rename(columns={"child": "y_id", "name": "y_name"})
    df["x_type"] = "effect/phenotype"
    df["x_source"] = "HPO"
    df["y_type"] = "effect/phenotype"
    df["y_source"] = "HPO"
    df["relation"] = "phenotype_phenotype"
    df["display_relation"] = "parent-child"
    return clean_edges(df)


def build_edge_disease_phenotype(
    hpoa_df: pd.DataFrame,
    mondo_xref: pd.DataFrame,
    hp_terms: pd.DataFrame,
    mondo_terms: pd.DataFrame,
    positive: bool = True,
) -> pd.DataFrame:
    """Build disease-phenotype edges from HPO-A annotations."""
    df = pd.merge(hpoa_df, mondo_xref, "left", left_on="disease_ontology_id", right_on="ontology_id")
    # Match ontology types (OMIM->OMIM, ORPHA->Orphanet, etc.)
    df = df.query('(disease_ontology==ontology) or (disease_ontology=="ORPHA" and ontology=="Orphanet")')
    df = pd.merge(df, hp_terms, "left", left_on="hp_id", right_on="id").rename(columns={"name": "hp_name"})
    df = pd.merge(df, mondo_terms, "left", left_on="mondo_id", right_on="id").rename(columns={"name": "mondo_name"})
    df = df.rename(columns={"mondo_id": "x_id", "mondo_name": "x_name", "hp_id": "y_id", "hp_name": "y_name"})
    df["x_source"] = "MONDO"
    df["x_type"] = "disease"
    df["y_source"] = "HPO"
    df["y_type"] = "effect/phenotype"
    relation = "disease_phenotype_positive" if positive else "disease_phenotype_negative"
    display = "phenotype present" if positive else "phenotype absent"
    df["relation"] = relation
    df["display_relation"] = display
    return clean_edges(df)


def build_edge_phenotype_protein(
    disgenet: pd.DataFrame,
    hp_xref: pd.DataFrame,
    hp_terms: pd.DataFrame,
) -> pd.DataFrame:
    """Build phenotype-protein edges from DisGeNET phenotypes via HPO cross-refs."""
    df = disgenet.query('diseaseType=="phenotype"').copy()
    df["geneId"] = df["geneId"].astype(int).astype(str)
    df = pd.merge(df, hp_xref, "inner", left_on="diseaseId", right_on="ontology_id")
    df = pd.merge(df, hp_terms, "left", left_on="hp_id", right_on="id")
    df = df.rename(columns={"geneId": "x_id", "geneSymbol": "x_name", "hp_id": "y_id", "name": "y_name"})
    df["x_type"] = "gene/protein"
    df["x_source"] = "NCBI"
    df["y_type"] = "effect/phenotype"
    df["y_source"] = "HPO"
    df["relation"] = "phenotype_protein"
    df["display_relation"] = "associated with"
    return clean_edges(df)


def build_edge_disease_protein(
    disgenet: pd.DataFrame,
    umls_mondo: pd.DataFrame,
    mondo_terms: pd.DataFrame,
) -> pd.DataFrame:
    """Build disease-protein edges from DisGeNET diseases via UMLS->MONDO mapping."""
    df = disgenet.query('diseaseType=="disease"').copy()
    df["geneId"] = df["geneId"].astype(int).astype(str)
    df = pd.merge(df, umls_mondo, "inner", left_on="diseaseId", right_on="umls_id")
    df = pd.merge(df, mondo_terms, "left", left_on="mondo_id", right_on="id")
    df = df.rename(columns={"geneId": "x_id", "geneSymbol": "x_name", "mondo_id": "y_id", "name": "y_name"})
    df["x_type"] = "gene/protein"
    df["x_source"] = "NCBI"
    df["y_type"] = "disease"
    df["y_source"] = "MONDO"
    df["relation"] = "disease_protein"
    df["display_relation"] = "associated with"
    return clean_edges(df)


def build_umls_mondo_mapping(mondo_xref: pd.DataFrame) -> pd.DataFrame:
    """Build UMLS->MONDO mapping from MONDO cross-references only.

    This is the 'direct mapping' from map_umls_mondo.py line 9.
    Doesn't require the full UMLS Metathesaurus.
    """
    direct = mondo_xref.query('ontology=="UMLS"')[["ontology_id", "mondo_id"]].rename(
        columns={"ontology_id": "umls_id"}
    ).drop_duplicates()
    logger.info(f"  Built {len(direct)} direct UMLS->MONDO mappings from MONDO cross-refs")
    return direct


def build_edge_go_go(
    go_terms: pd.DataFrame,
    go_edges: pd.DataFrame,
    go_type: str,
) -> pd.DataFrame:
    """Build GO term hierarchy edges for a specific GO type."""
    type_terms = go_terms.query(f'go_term_type=="{go_type}"')
    df = pd.merge(go_edges, type_terms, "inner", left_on="x", right_on="go_term_id")
    df = df.rename(columns={"go_term_id": "x_id", "go_term_name": "x_name", "go_term_type": "x_type"})
    df = pd.merge(df, type_terms, "inner", left_on="y", right_on="go_term_id")
    df = df.rename(columns={"go_term_id": "y_id", "go_term_name": "y_name", "go_term_type": "y_type"})

    relation_map = {
        "biological_process": "bioprocess_bioprocess",
        "molecular_function": "molfunc_molfunc",
        "cellular_component": "cellcomp_cellcomp",
    }
    df["relation"] = relation_map[go_type]
    df["x_source"] = "GO"
    df["y_source"] = "GO"
    df["display_relation"] = "parent-child"
    return clean_edges(df)


def build_edge_go_protein(
    gene2go: pd.DataFrame,
    go_terms: pd.DataFrame,
    gene_names: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Build protein-GO edges. Returns list of [molfunc, cellcomp, bioprocess] DataFrames."""
    gene2go = gene2go.copy()
    gene2go["ncbi_gene_id"] = gene2go["ncbi_gene_id"].astype(str)
    df = pd.merge(gene2go, go_terms, "inner", on="go_term_id")
    if "go_term_type_x" in df.columns:
        df = df.rename(columns={"go_term_type_x": "go_term_type"})
    df = pd.merge(df, gene_names, "left", left_on="ncbi_gene_id", right_on="ncbi_id")
    df = df.rename(columns={
        "ncbi_gene_id": "x_id", "symbol": "x_name",
        "go_term_id": "y_id", "go_term_name": "y_name", "go_term_type": "y_type",
    })
    df["x_type"] = "gene/protein"
    df["x_source"] = "NCBI"
    df["y_source"] = "GO"
    df["display_relation"] = "interacts with"

    relation_map = {
        "molecular_function": "molfunc_protein",
        "cellular_component": "cellcomp_protein",
        "biological_process": "bioprocess_protein",
    }

    results = []
    for go_type, relation in relation_map.items():
        sub = df.query(f'y_type=="{go_type}"').copy()
        sub["relation"] = relation
        results.append(clean_edges(sub))
    return results


def build_edge_exposure_protein(
    exposures: pd.DataFrame,
    gene_names: pd.DataFrame,
) -> pd.DataFrame:
    """Build exposure-protein edges from CTD."""
    cols = ["exposurestressorname", "exposurestressorid", "exposuremarker", "exposuremarkerid"]
    # Column names may vary by CTD version - try lowercase
    renames = {}
    for c in exposures.columns:
        for target in cols:
            if c.lower() == target.lower():
                renames[c] = target
    df = exposures.rename(columns=renames)[cols].copy()
    df = df.dropna(subset=["exposuremarkerid"])

    # Keep only rows where marker ID is numeric (gene ID)
    mask = df["exposuremarkerid"].astype(str).str.isnumeric()
    df = df[mask].copy()
    df["exposuremarkerid"] = df["exposuremarkerid"].astype(int).astype(str)
    df = pd.merge(df, gene_names, "left", left_on="exposuremarkerid", right_on="ncbi_id")
    df = df.rename(columns={
        "exposurestressorid": "x_id", "exposurestressorname": "x_name",
        "ncbi_id": "y_id", "symbol": "y_name",
    })
    df["x_type"] = "exposure"
    df["x_source"] = "CTD"
    df["y_type"] = "gene/protein"
    df["y_source"] = "NCBI"
    df["relation"] = "exposure_protein"
    df["display_relation"] = "interacts with"
    return clean_edges(df)


def build_edge_exposure_disease(
    exposures: pd.DataFrame,
    mondo_xref: pd.DataFrame,
    mondo_terms: pd.DataFrame,
) -> pd.DataFrame:
    """Build exposure-disease edges from CTD via MONDO MESH cross-refs."""
    df = exposures[["exposurestressorname", "exposurestressorid", "diseasename", "diseaseid"]].copy()
    df = df.dropna(subset=["diseaseid"])
    mesh_xref = mondo_xref.query('ontology=="MESH"')
    df = pd.merge(df, mesh_xref, "left", left_on="diseaseid", right_on="ontology_id")
    df = pd.merge(df, mondo_terms, "left", left_on="mondo_id", right_on="id")
    df = df.rename(columns={
        "exposurestressorid": "x_id", "exposurestressorname": "x_name",
        "mondo_id": "y_id", "name": "y_name",
    })
    df["x_type"] = "exposure"
    df["x_source"] = "CTD"
    df["y_type"] = "disease"
    df["y_source"] = "MONDO"
    df["relation"] = "exposure_disease"
    df["display_relation"] = "linked to"
    return clean_edges(df)


def build_edge_exposure_exposure(exposures: pd.DataFrame) -> pd.DataFrame:
    """Build exposure-exposure edges from CTD."""
    all_exp_ids = set(exposures["exposurestressorid"].dropna().unique())
    df = exposures[["exposurestressorname", "exposurestressorid", "exposuremarker", "exposuremarkerid"]].copy()
    df = df.dropna(subset=["exposuremarkerid"])
    df = df[df["exposuremarkerid"].isin(all_exp_ids)].drop_duplicates()
    df = df.rename(columns={
        "exposurestressorid": "x_id", "exposurestressorname": "x_name",
        "exposuremarker": "y_name", "exposuremarkerid": "y_id",
    })
    df["x_type"] = "exposure"
    df["x_source"] = "CTD"
    df["y_type"] = "exposure"
    df["y_source"] = "CTD"
    df["relation"] = "exposure_exposure"
    df["display_relation"] = "parent-child"
    return clean_edges(df)


def build_edge_exposure_go(
    exposures: pd.DataFrame,
    go_terms: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Build exposure-GO edges from CTD. Returns [bioprocess, molfunc, cellcomp]."""
    df = exposures[["exposurestressorname", "exposurestressorid", "phenotypename", "phenotypeid"]].copy()
    df = df.dropna(subset=["phenotypeid"])
    # Only keep GO-formatted IDs (e.g. "GO:0006915")
    df = df[df["phenotypeid"].str.contains(":", na=False)]
    df["phenotypeid"] = [str(int(x.split(":")[1])) for x in df["phenotypeid"].values]
    df = df.drop_duplicates()
    df = pd.merge(df, go_terms, "inner", left_on="phenotypeid", right_on="go_term_id")
    df = df.rename(columns={
        "exposurestressorid": "x_id", "exposurestressorname": "x_name",
        "go_term_id": "y_id", "go_term_name": "y_name", "go_term_type": "y_type",
    })
    df["x_type"] = "exposure"
    df["x_source"] = "CTD"
    df["y_source"] = "GO"

    relation_map = {
        "biological_process": "exposure_bioprocess",
        "molecular_function": "exposure_molfunc",
        "cellular_component": "exposure_cellcomp",
    }

    results = []
    for go_type, relation in relation_map.items():
        sub = df.query(f'y_type=="{go_type}"').copy()
        sub["relation"] = relation
        sub["display_relation"] = "interacts with"
        results.append(clean_edges(sub))
    return results


def build_edge_anatomy_anatomy(
    uberon_terms: pd.DataFrame,
    uberon_is_a: pd.DataFrame,
) -> pd.DataFrame:
    """Build anatomy-anatomy edges from UBERON hierarchy."""
    df = pd.merge(uberon_is_a, uberon_terms, "left", left_on="id", right_on="id")
    df = df.rename(columns={"id": "x_id", "name": "x_name"})
    df = pd.merge(df, uberon_terms, "left", left_on="is_a", right_on="id")
    df = df.rename(columns={"id": "y_id", "name": "y_name"})
    df["x_type"] = "anatomy"
    df["x_source"] = "UBERON"
    df["y_type"] = "anatomy"
    df["y_source"] = "UBERON"
    df["relation"] = "anatomy_anatomy"
    df["display_relation"] = "parent-child"
    return clean_edges(df)


def build_edge_anatomy_protein(
    bgee: pd.DataFrame,
    gene_names: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build anatomy-protein edges from Bgee. Returns (present, absent)."""
    df = pd.merge(bgee, gene_names, "inner", left_on="gene_name", right_on="symbol")
    df = df.rename(columns={
        "ncbi_id": "x_id", "symbol": "x_name",
        "anatomy_id": "y_id", "anatomy_name": "y_name",
    })
    df["x_source"] = "NCBI"
    df["x_type"] = "gene/protein"
    df["y_source"] = "UBERON"
    df["y_type"] = "anatomy"

    df_pos = df.query('expression=="present"').copy()
    df_pos["relation"] = "anatomy_protein_present"
    df_pos["display_relation"] = "expression present"

    df_neg = df.query('expression=="absent"').copy()
    df_neg["relation"] = "anatomy_protein_absent"
    df_neg["display_relation"] = "expression absent"

    return clean_edges(df_pos), clean_edges(df_neg)


def build_edge_pathway_pathway(
    reactome_terms: pd.DataFrame,
    reactome_rels: pd.DataFrame,
) -> pd.DataFrame:
    """Build pathway-pathway edges from Reactome hierarchy."""
    df = pd.merge(reactome_rels, reactome_terms, "inner", left_on="reactome_id_1", right_on="reactome_id")
    df = df.rename(columns={"reactome_id": "x_id", "reactome_name": "x_name"})
    df = pd.merge(df, reactome_terms, "inner", left_on="reactome_id_2", right_on="reactome_id")
    df = df.rename(columns={"reactome_id": "y_id", "reactome_name": "y_name"})
    df["x_source"] = "REACTOME"
    df["x_type"] = "pathway"
    df["y_source"] = "REACTOME"
    df["y_type"] = "pathway"
    df["relation"] = "pathway_pathway"
    df["display_relation"] = "parent-child"
    return clean_edges(df)


def build_edge_pathway_protein(
    reactome_ncbi: pd.DataFrame,
    _reactome_terms: pd.DataFrame,
    gene_names: pd.DataFrame,
) -> pd.DataFrame:
    """Build pathway-protein edges from Reactome."""
    df = pd.merge(reactome_ncbi, gene_names, "inner", on="ncbi_id")
    df = df.rename(columns={
        "ncbi_id": "x_id", "symbol": "x_name",
        "reactome_id": "y_id", "reactome_name": "y_name",
    })
    df["x_source"] = "NCBI"
    df["x_type"] = "gene/protein"
    df["y_source"] = "REACTOME"
    df["y_type"] = "pathway"
    df["relation"] = "pathway_protein"
    df["display_relation"] = "interacts with"
    return clean_edges(df)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def process_all_sources(data_dir: str | Path, config: dict) -> dict[str, pd.DataFrame]:
    """Process all downloaded databases and return intermediate DataFrames."""
    data_dir = Path(data_dir)
    databases = config.get("databases", {})
    result = {}

    # Gene names (needed by many)
    if databases.get("gene_names", {}).get("enabled"):
        logger.info("Processing gene_names...")
        result["gene_names"] = load_gene_names(data_dir)

    if databases.get("mondo", {}).get("enabled"):
        logger.info("Processing MONDO...")
        result["mondo_terms"], result["mondo_parents"], result["mondo_xref"] = process_mondo(data_dir)

    if databases.get("hpo", {}).get("enabled"):
        logger.info("Processing HPO...")
        result["hp_terms"], result["hp_parents"], result["hp_xref"] = process_hpo(data_dir)

    if databases.get("hpoa", {}).get("enabled"):
        logger.info("Processing HPO-A...")
        result["hpoa_pos"], result["hpoa_neg"] = process_hpoa(data_dir)

    if databases.get("gene_ontology", {}).get("enabled"):
        logger.info("Processing Gene Ontology...")
        result["go_terms"], result["go_edges"] = process_go(data_dir)

    if databases.get("gene2go", {}).get("enabled"):
        logger.info("Processing Gene2GO...")
        result["gene2go"] = process_gene2go(data_dir)

    if databases.get("bgee", {}).get("enabled"):
        logger.info("Processing Bgee...")
        result["bgee"] = process_bgee(data_dir)

    if databases.get("ctd", {}).get("enabled"):
        logger.info("Processing CTD...")
        result["exposures"] = process_ctd(data_dir)

    if databases.get("reactome", {}).get("enabled"):
        logger.info("Processing Reactome...")
        result["reactome_terms"], result["reactome_rels"], result["reactome_ncbi"] = process_reactome(data_dir)

    if databases.get("uberon", {}).get("enabled"):
        logger.info("Processing UBERON...")
        result["uberon_terms"], result["uberon_rels"], result["uberon_is_a"] = process_uberon(data_dir)

    if databases.get("disgenet", {}).get("enabled"):
        logger.info("Loading DisGeNET...")
        dg_path = data_dir / "disgenet" / "curated_gene_disease_associations.tsv"
        result["disgenet"] = pd.read_csv(str(dg_path), sep="\t", low_memory=False)
        result["disgenet"]["geneId"] = result["disgenet"]["geneId"].astype(int).astype(str)

    return result


def build_kg(
    data_dir: str | Path,
    config: dict,
    carry_from: str | Path | None = None,
) -> pd.DataFrame:
    """Build the full KG from processed source data.

    Args:
        data_dir: Directory with downloaded/processed source data.
        config: Configuration dict with enabled databases.
        carry_from: Path to an existing KG CSV to carry drug edges from.

    Returns:
        Complete KG DataFrame with standard PrimeKG columns.
    """
    data_dir = Path(data_dir)
    databases = config.get("databases", {})

    # Process all sources
    proc = process_all_sources(data_dir, config)

    edge_dfs = []

    def _add(name, df):
        if df is not None and len(df) > 0:
            edge_dfs.append(df)
            logger.info(f"  {name}: {len(df):,} edges")

    # --- Ontology hierarchies ---
    if "mondo_terms" in proc and "mondo_parents" in proc:
        _add("disease_disease", build_edge_disease_disease(proc["mondo_terms"], proc["mondo_parents"]))

    if "hp_terms" in proc and "hp_parents" in proc:
        _add("phenotype_phenotype", build_edge_phenotype_phenotype(proc["hp_terms"], proc["hp_parents"]))

    # --- Disease-phenotype (HPO-A) ---
    if all(k in proc for k in ["hpoa_pos", "mondo_xref", "hp_terms", "mondo_terms"]):
        _add("disease_phenotype_positive",
             build_edge_disease_phenotype(proc["hpoa_pos"], proc["mondo_xref"],
                                          proc["hp_terms"], proc["mondo_terms"], positive=True))
        if "hpoa_neg" in proc:
            _add("disease_phenotype_negative",
                 build_edge_disease_phenotype(proc["hpoa_neg"], proc["mondo_xref"],
                                              proc["hp_terms"], proc["mondo_terms"], positive=False))

    # --- DisGeNET: phenotype_protein and disease_protein ---
    if "disgenet" in proc:
        if "hp_xref" in proc and "hp_terms" in proc:
            _add("phenotype_protein",
                 build_edge_phenotype_protein(proc["disgenet"], proc["hp_xref"], proc["hp_terms"]))

        if "mondo_xref" in proc and "mondo_terms" in proc:
            # Build UMLS->MONDO mapping from MONDO cross-refs (no UMLS license needed)
            umls_mondo = build_umls_mondo_mapping(proc["mondo_xref"])
            if len(umls_mondo) > 0:
                _add("disease_protein",
                     build_edge_disease_protein(proc["disgenet"], umls_mondo, proc["mondo_terms"]))

    # --- GO hierarchies ---
    if "go_terms" in proc and "go_edges" in proc:
        for go_type in ["biological_process", "molecular_function", "cellular_component"]:
            _add(f"go_{go_type}", build_edge_go_go(proc["go_terms"], proc["go_edges"], go_type))

    # --- Gene-GO annotations ---
    if all(k in proc for k in ["gene2go", "go_terms", "gene_names"]):
        for df in build_edge_go_protein(proc["gene2go"], proc["go_terms"], proc["gene_names"]):
            _add(f"go_protein_{len(edge_dfs)}", df)

    # --- Exposure edges (CTD) ---
    if "exposures" in proc:
        if "gene_names" in proc:
            _add("exposure_protein", build_edge_exposure_protein(proc["exposures"], proc["gene_names"]))
        if "mondo_xref" in proc and "mondo_terms" in proc:
            _add("exposure_disease", build_edge_exposure_disease(proc["exposures"], proc["mondo_xref"], proc["mondo_terms"]))
        _add("exposure_exposure", build_edge_exposure_exposure(proc["exposures"]))
        if "go_terms" in proc:
            for df in build_edge_exposure_go(proc["exposures"], proc["go_terms"]):
                _add(f"exposure_go_{len(edge_dfs)}", df)

    # --- Anatomy edges ---
    if "uberon_terms" in proc and "uberon_is_a" in proc:
        _add("anatomy_anatomy", build_edge_anatomy_anatomy(proc["uberon_terms"], proc["uberon_is_a"]))

    if "bgee" in proc and "gene_names" in proc:
        pos, neg = build_edge_anatomy_protein(proc["bgee"], proc["gene_names"])
        _add("anatomy_protein_present", pos)
        _add("anatomy_protein_absent", neg)

    # --- Pathway edges ---
    if "reactome_terms" in proc and "reactome_rels" in proc:
        _add("pathway_pathway", build_edge_pathway_pathway(proc["reactome_terms"], proc["reactome_rels"]))

    if all(k in proc for k in ["reactome_ncbi", "reactome_terms", "gene_names"]):
        _add("pathway_protein",
             build_edge_pathway_protein(proc["reactome_ncbi"], proc["reactome_terms"], proc["gene_names"]))

    # --- Carry edges from existing KG for disabled databases ---
    if carry_from:
        carry_from = Path(carry_from)
        if carry_from.exists():
            logger.info(f"Carrying edges from {carry_from} for disabled databases...")
            existing = pd.read_csv(str(carry_from), low_memory=False)
            # Ensure standard columns exist
            for col in EDGE_COLUMNS:
                if col not in existing.columns:
                    existing[col] = ""

            # Carry drug edges (DrugBank, Drug Central, SIDER)
            if not databases.get("drugbank", {}).get("enabled"):
                drug_edges = existing[
                    (existing["x_type"] == "drug") | (existing["y_type"] == "drug")
                ][EDGE_COLUMNS].drop_duplicates()
                if len(drug_edges) > 0:
                    _add("carried_drug_edges", drug_edges)

            # Carry PPI edges
            if not databases.get("ppi", {}).get("enabled"):
                ppi_edges = existing[existing["relation"] == "protein_protein"][EDGE_COLUMNS].drop_duplicates()
                if len(ppi_edges) > 0:
                    _add("carried_ppi_edges", ppi_edges)

            # Carry DisGeNET edges (disease_protein, phenotype_protein)
            if not databases.get("disgenet", {}).get("enabled"):
                disgenet_rels = {"disease_protein", "phenotype_protein"}
                dg_edges = existing[existing["relation"].isin(disgenet_rels)][EDGE_COLUMNS].drop_duplicates()
                if len(dg_edges) > 0:
                    _add("carried_disgenet_edges", dg_edges)

            # Carry Reactome edges (pathway_pathway, pathway_protein)
            if not databases.get("reactome", {}).get("enabled"):
                react_rels = {"pathway_pathway", "pathway_protein"}
                react_edges = existing[existing["relation"].isin(react_rels)][EDGE_COLUMNS].drop_duplicates()
                if len(react_edges) > 0:
                    _add("carried_reactome_edges", react_edges)

    if not edge_dfs:
        logger.error("No edges built! Check that databases are enabled and data is downloaded.")
        return pd.DataFrame(columns=EDGE_COLUMNS)

    # Concatenate all edges
    logger.info("Assembling final KG...")
    kg = pd.concat(edge_dfs, ignore_index=True)
    kg = kg.drop_duplicates()

    # Add reverse edges (PrimeKG stores both directions)
    kg_rev = kg.rename(columns={
        "x_id": "y_id", "x_type": "y_type", "x_name": "y_name", "x_source": "y_source",
        "y_id": "x_id", "y_type": "x_type", "y_name": "x_name", "y_source": "x_source",
    })
    kg = pd.concat([kg, kg_rev], ignore_index=True)
    kg = kg.drop_duplicates().dropna()

    # Ensure all IDs are strings
    kg["x_id"] = kg["x_id"].astype(str)
    kg["y_id"] = kg["y_id"].astype(str)

    # Add node indices
    nodes = pd.concat([
        kg[["x_id", "x_type", "x_name", "x_source"]].rename(
            columns={"x_id": "node_id", "x_type": "node_type", "x_name": "node_name", "x_source": "node_source"}),
        kg[["y_id", "y_type", "y_name", "y_source"]].rename(
            columns={"y_id": "node_id", "y_type": "node_type", "y_name": "node_name", "y_source": "node_source"}),
    ]).drop_duplicates().reset_index(drop=True)
    nodes.index.name = "node_index"
    nodes = nodes.reset_index()

    kg = pd.merge(kg, nodes, "left",
                  left_on=["x_id", "x_type", "x_name", "x_source"],
                  right_on=["node_id", "node_type", "node_name", "node_source"])
    kg = kg.rename(columns={"node_index": "x_index"}).drop(
        columns=["node_id", "node_type", "node_name", "node_source"])

    kg = pd.merge(kg, nodes, "left",
                  left_on=["y_id", "y_type", "y_name", "y_source"],
                  right_on=["node_id", "node_type", "node_name", "node_source"])
    kg = kg.rename(columns={"node_index": "y_index"}).drop(
        columns=["node_id", "node_type", "node_name", "node_source"])

    # Final column order matching PrimeKG
    final_cols = ["relation", "display_relation", "x_index", "x_id", "x_type", "x_name", "x_source",
                  "y_index", "y_id", "y_type", "y_name", "y_source"]
    for col in final_cols:
        if col not in kg.columns:
            kg[col] = ""
    kg = kg[final_cols]

    logger.info(f"Final KG: {len(kg):,} edges, {len(nodes):,} nodes, "
                f"{kg['relation'].nunique()} relation types")

    return kg
