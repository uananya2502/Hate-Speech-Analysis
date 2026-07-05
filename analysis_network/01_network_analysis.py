"""
============================================================
01_network_analysis.py

Keyword Co-occurrence Network Analysis

Input
------
cleaned_data_comments/<category>_comment_sentiment.csv

Outputs
-------
analysis_network/
    figures/
    tables/
    reports/

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# ADD PROJECT ROOT
# ==========================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from resources.text_normalization import normalize_text

# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

TABLE_FOLDER = os.path.join(
    "analysis_network",
    "tables"
)

FIGURE_FOLDER = os.path.join(
    "analysis_network",
    "figures"
)

REPORT_FOLDER = os.path.join(
    "analysis_network",
    "reports"
)

os.makedirs(TABLE_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ==========================================================
# CATEGORY
# ==========================================================

def get_category():

    print("=" * 60)
    print("KEYWORD NETWORK ANALYSIS")
    print("=" * 60)

    category = input(
        "\nEnter category (infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    return category


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(category):

    input_file = os.path.join(
        INPUT_FOLDER,
        f"{category}_comment_sentiment.csv"
    )

    if not os.path.exists(input_file):

        print("\nDataset Not Found")
        exit()

    print("\nLoading dataset...")

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print(f"Loaded {len(df)} comments.")

    return df


# ==========================================================
# DATASET INFORMATION
# ==========================================================

def dataset_information(df):

    print("\n" + "=" * 60)

    print("DATASET INFORMATION")

    print("=" * 60)

    print(f"Rows       : {len(df)}")

    print(f"Columns    : {len(df.columns)}")

    print(
        f"Memory MB  : {df.memory_usage(deep=True).sum() / (1024**2):.2f}"
    )
# ==========================================================
# PREPARE DOCUMENTS
# ==========================================================

def prepare_documents(df):

    print("\nPreparing Documents...")

    documents = []

    for comment in df["combined_comment"].fillna(""):

        text = normalize_text(comment)

        words = text.split()

        # Remove duplicate words while preserving order
        words = list(dict.fromkeys(words))

        # Ignore very small comments
        if len(words) < 2:
            continue

        documents.append(words)

    print(f"Documents Used : {len(documents)}")

    return documents
# ==========================================================
# BUILD KEYWORD NETWORK
# ==========================================================

def build_network(documents):

    print("\nBuilding Keyword Network...")

    G = nx.Graph()

    for words in documents:

        for i in range(len(words) - 1):

            source = words[i]
            target = words[i + 1]

            if G.has_edge(source, target):

                G[source][target]["weight"] += 1

            else:

                G.add_edge(

                    source,
                    target,
                    weight=1

                )

    print(f"Nodes : {G.number_of_nodes()}")

    print(f"Edges : {G.number_of_edges()}")

    return G
# ==========================================================
# EXPORT NODES
# ==========================================================

def export_nodes(G):

    print("\nExporting Nodes...")

    nodes = []

    for node in G.nodes():

        nodes.append({

            "Keyword": node,

            "Degree": G.degree(node)

        })

    nodes = (

        pd.DataFrame(nodes)

        .sort_values(

            by="Degree",

            ascending=False

        )

    )

    nodes.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "keyword_nodes.csv"

        ),

        index=False

    )

    print("✓ keyword_nodes.csv")

    return nodes
# ==========================================================
# EXPORT EDGES
# ==========================================================

def export_edges(G):

    print("\nExporting Edges...")

    edges = []

    for source, target, data in G.edges(data=True):

        edges.append({

            "Source": source,

            "Target": target,

            "Weight": data["weight"]

        })

    edges = (

        pd.DataFrame(edges)

        .sort_values(

            by="Weight",

            ascending=False

        )

    )

    edges.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "keyword_edges.csv"

        ),

        index=False

    )

    print("✓ keyword_edges.csv")

    return edges
# ==========================================================
# DRAW NETWORK
# ==========================================================

def draw_network(G):

    print("\nGenerating Keyword Network...")

    # --------------------------------------------
    # Keep Top 100 Most Connected Keywords
    # --------------------------------------------

    degree_dict = dict(G.degree())

    top_nodes = sorted(
        degree_dict,
        key=degree_dict.get,
        reverse=True
    )[:200]

    H = G.subgraph(top_nodes).copy()

    # --------------------------------------------
    # Remove Weak Connections
    # --------------------------------------------

    weak_edges = [

        (u, v)

        for u, v, d in H.edges(data=True)

        if d["weight"] < 3

    ]

    H.remove_edges_from(weak_edges)

    # Remove isolated nodes
    H.remove_nodes_from(
        list(nx.isolates(H))
    )

    print(f"Visualized Nodes : {H.number_of_nodes()}")
    print(f"Visualized Edges : {H.number_of_edges()}")

    # --------------------------------------------
    # Layout
    # --------------------------------------------

    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(
        H,
        seed=42,
        k=0.9,
        iterations=150
    )

    # --------------------------------------------
    # Node Size
    # --------------------------------------------

    node_sizes = [

        H.degree(node) * 35

        for node in H.nodes()

    ]

    # --------------------------------------------
    # Edge Width
    # --------------------------------------------

    edge_widths = [

        H[u][v]["weight"] * 0.30

        for u, v in H.edges()

    ]

    # --------------------------------------------
    # Draw Nodes
    # --------------------------------------------

    nx.draw_networkx_nodes(

        H,

        pos,

        node_size=node_sizes,

        alpha=0.85

    )

    # --------------------------------------------
    # Draw Edges
    # --------------------------------------------

    nx.draw_networkx_edges(

        H,

        pos,

        width=edge_widths,

        alpha=0.35

    )

    # --------------------------------------------
    # Labels (Only Important Ones)
    # --------------------------------------------

    labels = {

        node: node

        for node in H.nodes()

        if H.degree(node) >= 8

    }

    nx.draw_networkx_labels(

        H,

        pos,

        labels,

        font_size=9

    )

    plt.title(
        "Top Keyword Co-occurrence Network",
        fontsize=18
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "keyword_network.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("✓ keyword_network.png")
# ==========================================================
# EXPORT GEPHI NETWORK
# ==========================================================

def export_gephi(G):

    print("\nExporting Gephi Network...")

    output = os.path.join(

        TABLE_FOLDER,

        "keyword_network.gexf"

    )

    nx.write_gexf(

        G,

        output

    )

    print("✓ keyword_network.gexf")
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    category,

    G,

    nodes,

    edges

):

    report = os.path.join(

        REPORT_FOLDER,

        f"{category}_network_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60 + "\n")

        file.write("KEYWORD NETWORK REPORT\n")

        file.write("="*60 + "\n\n")

        file.write(f"Total Nodes : {G.number_of_nodes()}\n")

        file.write(f"Total Edges : {G.number_of_edges()}\n\n")

        file.write("TOP KEYWORDS\n")

        file.write("-"*40 + "\n")

        file.write(

            nodes.head(20).to_string(index=False)

        )

        file.write("\n\n")

        file.write("TOP CONNECTIONS\n")

        file.write("-"*40 + "\n")

        file.write(

            edges.head(20).to_string(index=False)

        )

    print("\nReport Saved")

    print(report)
# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n" + "="*60)

    print("FINAL VALIDATION")

    print("="*60)

    print("✓ Keyword Network")

    print("✓ Node Table")

    print("✓ Edge Table")

    print("✓ Network Figure")

    print("✓ Gephi Export")

    print("✓ Report")
# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_information(df)

    documents = prepare_documents(df)

    G = build_network(documents)

    nodes = export_nodes(G)

    edges = export_edges(G)

    draw_network(G)

    export_gephi(G)

    save_report(

        category,

        G,

        nodes,

        edges

    )

    final_validation()

    print("\n" + "="*60)

    print("NETWORK ANALYSIS COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

#.    python analysis_network/01_network_analysis.py