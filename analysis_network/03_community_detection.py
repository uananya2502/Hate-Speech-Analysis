"""
============================================================
03_community_detection.py

Community Detection

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

from networkx.algorithms.community import (
    greedy_modularity_communities
)

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

TOP_NODES = 200


# ==========================================================
# LOAD NETWORK
# ==========================================================

def load_network():

    print("=" * 60)
    print("COMMUNITY DETECTION")
    print("=" * 60)

    network_file = os.path.join(
        TABLE_FOLDER,
        "keyword_network.gexf"
    )

    if not os.path.exists(network_file):

        print("\nNetwork File Not Found")
        exit()

    print("\nLoading Network...")

    G = nx.read_gexf(network_file)

    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")

    return G
# ==========================================================
# FILTER NETWORK
# ==========================================================

def filter_network(G):

    print(f"\nFiltering Network (Top {TOP_NODES} Nodes)...")

    degree_dict = dict(G.degree())

    top_nodes = sorted(
        degree_dict,
        key=degree_dict.get,
        reverse=True
    )[:TOP_NODES]

    H = G.subgraph(top_nodes).copy()

    print(f"Filtered Nodes : {H.number_of_nodes()}")
    print(f"Filtered Edges : {H.number_of_edges()}")

    return H
# ==========================================================
# COMMUNITY DETECTION
# ==========================================================

def detect_communities(G):

    print("\nDetecting Communities...")

    communities = list(

        greedy_modularity_communities(

            G,

            weight="weight"

        )

    )

    print(f"Communities Found : {len(communities)}")

    return communities
# ==========================================================
# SAVE COMMUNITY TABLES
# ==========================================================

def save_communities(communities):

    print("\nSaving Community Tables...")

    rows = []

    summary = []

    for i, community in enumerate(communities, start=1):

        words = sorted(list(community))

        summary.append({

            "Community": i,

            "Keywords": len(words)

        })

        for word in words:

            rows.append({

                "Community": i,

                "Keyword": word

            })

    community_df = pd.DataFrame(rows)

    summary_df = pd.DataFrame(summary)

    community_df.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "communities.csv"

        ),

        index=False

    )

    summary_df.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "community_summary.csv"

        ),

        index=False

    )

    print("✓ communities.csv")

    print("✓ community_summary.csv")

    return community_df, summary_df
# ==========================================================
# PLOT COMMUNITIES
# ==========================================================

def plot_communities(G, communities):

    print("\nGenerating Community Network...")

    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.9
    )

    # Draw edges first
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.25,
        width=0.8
    )

    # Different color for each community
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "gold",
        "brown",
        "pink"
    ]

    for i, community in enumerate(communities):

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(community),
            node_size=300,
            node_color=colors[i % len(colors)],
            alpha=0.9,
            label=f"Community {i+1}"
        )

    # Show labels only for important nodes
    labels = {}

    degree = dict(G.degree())

    for node in G.nodes():

        if degree[node] >= 15:

            labels[node] = node

    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        font_size=8
    )

    plt.title(
        "Keyword Communities",
        fontsize=18
    )

    plt.legend()

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "community_network.png"
        ),
        dpi=300
    )

    plt.close()

    print("✓ community_network.png")
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(summary_df, community_df):

    report_file = os.path.join(
        REPORT_FOLDER,
        "community_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 60 + "\n")
        file.write("COMMUNITY DETECTION REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Total Communities : {len(summary_df)}\n\n")

        file.write("COMMUNITY SUMMARY\n")
        file.write("-" * 40 + "\n")
        file.write(summary_df.to_string(index=False))

        file.write("\n\n")

        file.write("TOP KEYWORDS IN EACH COMMUNITY\n")
        file.write("-" * 40 + "\n\n")

        for community in summary_df["Community"]:

            words = community_df[
                community_df["Community"] == community
            ]["Keyword"].head(20).tolist()

            file.write(f"Community {community}\n")
            file.write(", ".join(words))
            file.write("\n\n")

    print("\n✓ community_report.txt")
# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n" + "=" * 60)

    print("FINAL VALIDATION")

    print("=" * 60)

    print("✓ Community Detection")

    print("✓ Community Tables")

    print("✓ Community Network")

    print("✓ Report")
# ==========================================================
# MAIN
# ==========================================================

def main():

    G = load_network()

    G = filter_network(G)

    communities = detect_communities(G)

    community_df, summary_df = save_communities(
        communities
    )

    plot_communities(
        G,
        communities
    )

    save_report(
        summary_df,
        community_df
    )

    final_validation()

    print("\n" + "=" * 60)

    print("COMMUNITY DETECTION COMPLETED")

    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()
#.   python analysis_network/03_community_detection.py