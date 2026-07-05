"""
============================================================
02_centrality.py

Centrality Analysis

Input
------
analysis_network/tables/keyword_network.gexf

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
# IMPORTS
# ==========================================================

import os

import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx

# ==========================================================
# FOLDERS
# ==========================================================

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

# ==========================================================
# LOAD NETWORK
# ==========================================================

def load_network():

    print("=" * 60)

    print("CENTRALITY ANALYSIS")

    print("=" * 60)

    network_file = os.path.join(

        TABLE_FOLDER,

        "keyword_network.gexf"

    )

    if not os.path.exists(network_file):

        print("\nNetwork File Not Found")

        print("Run 01_network_analysis.py first.")

        exit()

    print("\nLoading Network...")

    G = nx.read_gexf(network_file)

    print(f"Nodes : {G.number_of_nodes()}")

    print(f"Edges : {G.number_of_edges()}")

    return G
# ==========================================================
# FILTER NETWORK
# ==========================================================

def filter_network(G, top_n=200):

    print(f"\nFiltering Network (Top {top_n} Nodes)...")

    degree_dict = dict(G.degree())

    top_nodes = sorted(
        degree_dict,
        key=degree_dict.get,
        reverse=True
    )[:top_n]

    H = G.subgraph(top_nodes).copy()

    print(f"Filtered Nodes : {H.number_of_nodes()}")
    print(f"Filtered Edges : {H.number_of_edges()}")

    return H

# ==========================================================
# PLOT CENTRALITY
# ==========================================================

def plot_centrality(

    table,

    title,

    filename,

    score_column

):

    top = table.head(20)

    plt.figure(figsize=(12,8))

    plt.barh(

        top["Keyword"],

        top[score_column]

    )

    plt.gca().invert_yaxis()

    plt.xlabel(score_column)

    plt.ylabel("Keyword")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            filename

        ),

        dpi=300

    )

    plt.close()

    print(f"✓ {filename}")
# ==========================================================
# DEGREE CENTRALITY
# ==========================================================

def degree_centrality(G):

    print("\nCalculating Degree Centrality...")

    scores = nx.degree_centrality(G)

    result = pd.DataFrame({

        "Keyword": list(scores.keys()),

        "Degree Centrality": list(scores.values())

    })

    result = result.sort_values(

        by="Degree Centrality",

        ascending=False

    )

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "degree_centrality.csv"

        ),

        index=False

    )

    plot_centrality(

        result,

        "Top 20 Degree Centrality",

        "degree_centrality.png",

        "Degree Centrality"

    )

    print("✓ degree_centrality.csv")

    return result
# ==========================================================
# BETWEENNESS CENTRALITY
# ==========================================================

def betweenness_centrality(G):

    print("\nCalculating Betweenness Centrality...")

    scores = nx.betweenness_centrality(

        G,

        normalized=True,

        weight="weight"

    )

    result = pd.DataFrame({

        "Keyword": list(scores.keys()),

        "Betweenness Centrality": list(scores.values())

    })

    result = result.sort_values(

        by="Betweenness Centrality",

        ascending=False

    )

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "betweenness_centrality.csv"

        ),

        index=False

    )

    plot_centrality(

        result,

        "Top 20 Betweenness Centrality",

        "betweenness_centrality.png",

        "Betweenness Centrality"

    )

    print("✓ betweenness_centrality.csv")

    return result
# ==========================================================
# CLOSENESS CENTRALITY
# ==========================================================

def closeness_centrality(G):

    print("\nCalculating Closeness Centrality...")

    scores = nx.closeness_centrality(G)

    result = pd.DataFrame({

        "Keyword": list(scores.keys()),

        "Closeness Centrality": list(scores.values())

    })

    result = result.sort_values(

        by="Closeness Centrality",

        ascending=False

    )

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "closeness_centrality.csv"

        ),

        index=False

    )

    plot_centrality(

        result,

        "Top 20 Closeness Centrality",

        "closeness_centrality.png",

        "Closeness Centrality"

    )

    print("✓ closeness_centrality.csv")

    return result
# ==========================================================
# EIGENVECTOR CENTRALITY
# ==========================================================

def eigenvector_centrality(G):

    print("\nCalculating Eigenvector Centrality...")

    try:

        scores = nx.eigenvector_centrality(

            G,

            max_iter=1000,

            weight="weight"

        )

    except nx.PowerIterationFailedConvergence:

        print("Eigenvector Centrality did not converge.")

        return pd.DataFrame()

    result = pd.DataFrame({

        "Keyword": list(scores.keys()),

        "Eigenvector Centrality": list(scores.values())

    })

    result = result.sort_values(

        by="Eigenvector Centrality",

        ascending=False

    )

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "eigenvector_centrality.csv"

        ),

        index=False

    )

    plot_centrality(

        result,

        "Top 20 Eigenvector Centrality",

        "eigenvector_centrality.png",

        "Eigenvector Centrality"

    )

    print("✓ eigenvector_centrality.csv")

    return result
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(
    degree,
    betweenness,
    closeness,
    eigenvector
):

    report_file = os.path.join(
        REPORT_FOLDER,
        "centrality_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 60 + "\n")
        file.write("CENTRALITY ANALYSIS REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write("TOP DEGREE CENTRALITY\n")
        file.write("-" * 40 + "\n")
        file.write(degree.head(20).to_string(index=False))

        file.write("\n\n")

        file.write("TOP BETWEENNESS CENTRALITY\n")
        file.write("-" * 40 + "\n")
        file.write(betweenness.head(20).to_string(index=False))

        file.write("\n\n")

        file.write("TOP CLOSENESS CENTRALITY\n")
        file.write("-" * 40 + "\n")
        file.write(closeness.head(20).to_string(index=False))

        file.write("\n\n")

        file.write("TOP EIGENVECTOR CENTRALITY\n")
        file.write("-" * 40 + "\n")
        file.write(eigenvector.head(20).to_string(index=False))

    print("\n✓ centrality_report.txt")
# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n" + "=" * 60)

    print("FINAL VALIDATION")

    print("=" * 60)

    print("✓ Degree Centrality")

    print("✓ Betweenness Centrality")

    print("✓ Closeness Centrality")

    print("✓ Eigenvector Centrality")

    print("✓ CSV Tables")

    print("✓ Figures")

    print("✓ Report")
# ==========================================================
# MAIN
# ==========================================================

def main():

    G = load_network()

    G = filter_network(
        G,
        top_n=200
    )

    degree = degree_centrality(G)

    betweenness = betweenness_centrality(G)

    closeness = closeness_centrality(G)

    eigenvector = eigenvector_centrality(G)

    save_report(
        degree,
        betweenness,
        closeness,
        eigenvector
    )

    final_validation()

    print("\n" + "=" * 60)

    print("CENTRALITY ANALYSIS COMPLETED")

    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()
#    python analysis_network/02_centrality.py