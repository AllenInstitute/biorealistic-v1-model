# Evaluate the change of the weights due to the TensorFlow training
# %%
import network_utils as nu
import numpy as np
import pandas as pd
import polars as pl
from importlib import reload


# %% First, let's define the cell types

ctdf = pl.DataFrame(nu.get_cell_type_table().reset_index()).lazy()

# %% load the network
network_dir = "core"

nodes = nu.load_nodes_pl(network_dir)
edges_orig = nu.load_edges_pl(network_dir)
# edges_adjusted = nu.load_edges_pl(network_dir, appendix="_adjusted")
edges_trained = nu.load_edges_pl(network_dir, appendix="_checkpoint")


# %% let's define the cell types of the source and target edges
def merge_tables(edges, nodes, ctdf):
    source_table = edges.join(nodes, left_on="source_id", right_on="node_id").join(
        ctdf.select(["pop_name", "cell_type"]), on="pop_name"
    )
    all_table = source_table.join(
        nodes, left_on="target_id", right_on="node_id", suffix="_target"
    ).join(
        ctdf.select(["pop_name", "cell_type"]),
        left_on="pop_name_target",
        right_on="pop_name",
        suffix="_target",
    )
    return all_table


all_table_orig = merge_tables(edges_orig, nodes, ctdf)
# all_table_adjusted = merge_tables(edges_adjusted, nodes, ctdf)
all_table_trained = merge_tables(edges_trained, nodes, ctdf)

# concatenate all the tables, with a column indicating the source of the data
all_table_orig = all_table_orig.with_columns(pl.lit("orig").alias("source"))
# all_table_adjusted = all_table_adjusted.with_columns(pl.lit("adjusted").alias("source"))
all_table_trained = all_table_trained.with_columns(pl.lit("trained").alias("source"))

# all_tables = pl.concat([all_table_orig, all_table_adjusted, all_table_trained])
all_df = pl.concat([all_table_orig, all_table_trained])

# all_df.fetch()


# %% let's plot the distribution of the weights


import seaborn as sns
import polars as pl
import matplotlib.pyplot as plt


# Assuming all_table is your original DataFrame
# 1. Select specific columns
def plot_one(all_df):
    selected_df = all_df.select(
        ["cell_type", "syn_weight", "source", "pop_name_target"]
    )

    # 2. Gather every 10th row
    gathered_df = selected_df.gather_every(10)

    # 3. Apply the absolute value and logarithm base 10 to the `syn_weight` column
    transformed_df = gathered_df.with_columns(
        (pl.col("syn_weight").abs().log10()).alias("syn_weight_log10")
    )

    # 4. Convert to a Pandas DataFrame
    plt_df = transformed_df.collect().to_pandas()

    plt.rcParams["font.family"] = "monospace"

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=plt_df,
        x="cell_type",
        y="syn_weight_log10",
        hue="source",
        order=np.sort(plt_df["cell_type"].unique()),
    )

    # rotate the x-axis labels
    t = plt.xticks(rotation=90)

    # set to Menlo font

    plt.xlabel("Source cell_type")
    plt.ylabel("log10 synaptic weight")


cell_type_names = ctdf.collect().select("cell_type").to_series().to_numpy()
cell_type_names = np.unique(cell_type_names)

for cell_type in cell_type_names:
    cell_type_fn = cell_type.replace("/", "")
    plot_one(all_df.filter(pl.col("cell_type_target") == cell_type))
    plt.title(f"Target cell type: {cell_type}")
    plt.tight_layout()
    plt.savefig(f"figs/syn_weight_{cell_type_fn}.png", dpi=200)

# %% Now we learned that the log-normal distributions are well preserved.
# I would make a heatmap of all the connection changes.

gathered_df = all_df.gather_every(10)
group_by_df = gathered_df.group_by(["cell_type", "cell_type_target", "source"])
agg_df = group_by_df.agg(pl.col("syn_weight").abs().log10().median())
agg_df_c = agg_df.collect()


# %%
agg_p = agg_df_c.to_pandas()

n_elem = len(cell_type_names)
matrix = np.zeros((n_elem, n_elem))


# let's make a matrix of the weights
for i in range(n_elem):
    for j in range(n_elem):
        orig = agg_p.query(
            f"cell_type == '{cell_type_names[i]}'"
            + f" and cell_type_target == '{cell_type_names[j]}'"
            + " and source == 'orig'"
        )
        trained = agg_p.query(
            f"cell_type == '{cell_type_names[i]}'"
            + f" and cell_type_target == '{cell_type_names[j]}'"
            + " and source == 'trained'"
        )

        if len(orig) > 0 and len(trained) > 0:
            trained_val = trained["syn_weight"].values[0]
            orig_val = orig["syn_weight"].values[0]
            matrix[i, j] = trained_val - orig_val
        else:
            if len(orig) == 0:
                print(f"no orig data for {cell_type_names[i]} -> {cell_type_names[j]}")
            if len(trained) == 0:
                print(
                    f"no trained data for {cell_type_names[i]} -> {cell_type_names[j]}"
                )
            matrix[i, j] = np.nan


# %% create a matrix as df
matrix_df = pd.DataFrame(matrix, index=cell_type_names, columns=cell_type_names)

# plot with sns
plt.figure(figsize=(13, 13))
sns.heatmap(
    matrix_df, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f"
)

plt.xlabel("Target cell type")
plt.ylabel("Source cell type")

plt.title("Change of synaptic weights (log10) due to training")

line_locs = [1, 5, 9, 15]
for h in line_locs:
    plt.axhline(h, color="w", linewidth=2)
    plt.axvline(h, color="w", linewidth=2)

plt.axis("image")
plt.tight_layout()

plt.savefig("syn_weight_change_matrix.pdf")
