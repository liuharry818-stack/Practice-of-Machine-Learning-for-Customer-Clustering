import os
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "8")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "Dataset.xlsx"
#create the folder for output pics
OUTPUT_DIR = Path(__file__).resolve().parent / "pics"
OUTPUT_DIR.mkdir(exist_ok=True)

#Due to my RAM limit, I will sample 2,000 rows for each scenario. The random state is fixed for reproducibility
SAMPLE_SIZE = 2000
RANDOM_STATE = 42
K_VALUES = range(2, 7)
CONTINUOUS_COLUMNS = ["INCOME", "TRB"]
AGE_MIDPOINTS = {"18-24": 21, "25-29": 27, "30-34": 32}
AGGLOMERATIVE_METHOD = "ward"
DIVISIVE_METHOD = "divisive"
CLUSTERING_METHODS = (AGGLOMERATIVE_METHOD, DIVISIVE_METHOD)

#read the dataset and check for missing values and duplicates
df = pd.read_excel(DATA_PATH, sheet_name="Synthetic Data")
print(f"Full dataset shape: {df.shape}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(f"Duplicate rows: {int(df.duplicated().sum())}")

#For consistency, we still use both midpoints and one-hot encoding methods for hierarchical clustering
age_midpoint = df["AGE"].map(AGE_MIDPOINTS)
if age_midpoint.isna().any():
    unknown_age_groups = df.loc[age_midpoint.isna(), "AGE"].unique().tolist()
    raise ValueError(f"Unmapped AGE categories: {unknown_age_groups}")

X_midpoint = df.drop(columns="AGE").copy()
X_midpoint["AGE"] = age_midpoint

age_one_hot = pd.get_dummies(df["AGE"], prefix="AGE", dtype=int)
X_one_hot = pd.concat([df.drop(columns="AGE"), age_one_hot], axis=1)

#Still we use IQR rule for outliers
outlier_flags = pd.DataFrame(index=df.index)
for column in CONTINUOUS_COLUMNS:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_flags[column] = (df[column] < lower_bound) | (df[column] > upper_bound)

outlier_mask = outlier_flags.any(axis=1)
print("\nIQR outliers excluded:")
for column in CONTINUOUS_COLUMNS:
    print(f"{column}: {int(outlier_flags[column].sum())}")
print(f"Unique rows flagged: {int(outlier_mask.sum())}")

scenarios = {
    "Original": ~pd.Series(False, index=df.index),
    "IQR excluded": ~outlier_mask,
}
sampled_data = {}

#feature engineering, we use z-score scaler for the data, keep binary variables as what they were
def prepare_features(features: pd.DataFrame, scale_age: bool) -> pd.DataFrame:
    """Z-score continuous inputs and midpoint age; retain binary indicators."""
    columns_to_scale = list(CONTINUOUS_COLUMNS)
    if scale_age:
        columns_to_scale.append("AGE")

    prepared = features.astype(float).copy()
    scaler = StandardScaler()
    prepared[columns_to_scale] = scaler.fit_transform(prepared[columns_to_scale])
    return prepared

hierarchical_results = {}
for scenario_name, keep_mask in scenarios.items():
    scenario_df = df.loc[keep_mask]
    sample = scenario_df.sample(
        n=min(SAMPLE_SIZE, len(scenario_df)),
        random_state=RANDOM_STATE,
    )
    sampled_data[scenario_name] = sample.copy()
    print(
        f"\n{scenario_name}: sampled {len(sample)} rows "
        f"from {len(scenario_df)} available rows"
    )

    raw_encodings = {
        "Midpoint": (X_midpoint.loc[sample.index], True),
        "One-hot": (X_one_hot.loc[sample.index], False),
    }

    for encoding, (raw_features, scale_age) in raw_encodings.items():
        features = prepare_features(raw_features, scale_age=scale_age)
        for method in CLUSTERING_METHODS:
            tree = None
            if method == AGGLOMERATIVE_METHOD:
                tree = linkage(features.to_numpy(), method=method, metric="euclidean")
            labels_by_k = {}
            scores_by_k = {}

            print(f"\n{scenario_name} | {encoding} | {method}")
            for k in K_VALUES:
                if method == DIVISIVE_METHOD:
                    model = BisectingKMeans(
                        n_clusters=k,
                        init="k-means++",
                        n_init=10,
                        bisecting_strategy="biggest_inertia",
                        random_state=RANDOM_STATE,
                    )
                    labels = model.fit_predict(features)
                else:
                    labels = fcluster(tree, t=k, criterion="maxclust")

                cluster_count = len(np.unique(labels))
                if cluster_count < 2 or cluster_count >= len(labels):
                    raise ValueError(
                        f"{method} produced {cluster_count} clusters "
                        f"for requested k={k} ({scenario_name}, {encoding})."
                    )

                score = silhouette_score(features, labels, metric="euclidean")
                labels_by_k[k] = labels
                scores_by_k[k] = score
                sizes = pd.Series(labels).value_counts().sort_index()
                print(
                    f"k={k}, actual clusters={cluster_count}, "
                    f"silhouette={score:.4f}"
                )
                print(sizes.to_string())

            hierarchical_results[
                (scenario_name, encoding, method)
            ] = {
                "features": features,
                "tree": tree,
                "labels_by_k": labels_by_k,
                "silhouette_scores": scores_by_k,
            }

#We also need readable details for customer profiles for hierarchical clustering
binary_columns = [
    column
    for column in df.select_dtypes(include="number").columns
    if set(df[column].dropna().unique()).issubset({0, 1})
]
profile_rows = []
binary_driver_rows = []

for (scenario_name, encoding, method), result in hierarchical_results.items():
    sample_data = sampled_data[scenario_name].loc[result["features"].index].copy()
    overall_binary_rates = sample_data[binary_columns].mean()
    age_categories = sorted(sample_data["AGE"].dropna().unique())

    for k, labels in result["labels_by_k"].items():
        labeled_data = sample_data.copy()
        labeled_data["cluster"] = labels
        cluster_sizes = labeled_data["cluster"].value_counts().sort_index()
        age_shares = pd.crosstab(
            labeled_data["cluster"],
            labeled_data["AGE"],
            normalize="index",
        ).reindex(columns=age_categories, fill_value=0)
        binary_rates = labeled_data.groupby("cluster")[binary_columns].mean()
        continuous_summary = labeled_data.groupby("cluster").agg(
            INCOME_mean=("INCOME", "mean"),
            INCOME_median=("INCOME", "median"),
            TRB_mean=("TRB", "mean"),
            TRB_median=("TRB", "median"),
        )

        for cluster_id in cluster_sizes.index:
            row = {
                "scenario": scenario_name,
                "encoding": encoding,
                "method": method,
                "k": k,
                "cluster": int(cluster_id),
                "cluster_size": int(cluster_sizes.loc[cluster_id]),
                "cluster_share": cluster_sizes.loc[cluster_id] / len(labeled_data),
                **continuous_summary.loc[cluster_id].to_dict(),
            }
            row.update(
                {
                    f"AGE_{age}_share": age_shares.loc[cluster_id, age]
                    for age in age_categories
                }
            )
            row.update(
                {
                    f"{feature}_rate": binary_rates.loc[cluster_id, feature]
                    for feature in binary_columns
                }
            )
            profile_rows.append(row)

            for feature in binary_columns:
                cluster_rate = binary_rates.loc[cluster_id, feature]
                overall_rate = overall_binary_rates[feature]
                binary_driver_rows.append(
                    {
                        "scenario": scenario_name,
                        "encoding": encoding,
                        "method": method,
                        "k": k,
                        "cluster": int(cluster_id),
                        "feature": feature,
                        "cluster_rate": cluster_rate,
                        "overall_rate": overall_rate,
                        "difference_percentage_points": 100
                        * (cluster_rate - overall_rate),
                        "lift_vs_overall": (
                            cluster_rate / overall_rate
                            if overall_rate > 0
                            else np.nan
                        ),
                    }
                )

#profile output(we do not really need it but I just refer to Github suggestion)
cluster_profiles = pd.DataFrame(profile_rows)
binary_drivers = pd.DataFrame(binary_driver_rows)
profile_path = OUTPUT_DIR / "hierarchical_cluster_profiles.csv"
drivers_path = OUTPUT_DIR / "hierarchical_binary_feature_contrasts.csv"
cluster_profiles.to_csv(profile_path, index=False, encoding="utf-8-sig")
binary_drivers.to_csv(drivers_path, index=False, encoding="utf-8-sig")
print(f"\nSaved sampled cluster profiles: {profile_path}")
print(f"Saved sampled binary-feature contrasts: {drivers_path}")

#We plot silhouette scores compare Ward agglomerative and divisive results.
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for row, scenario_name in enumerate(scenarios):
    for column, encoding in enumerate(("Midpoint", "One-hot")):
        ax = axes[row, column]
        for method in CLUSTERING_METHODS:
            results = hierarchical_results[(scenario_name, encoding, method)]
            ax.plot(
                list(K_VALUES),
                [results["silhouette_scores"][k] for k in K_VALUES],
                marker="o",
                label=(
                    "Agglomerative (Ward)"
                    if method == AGGLOMERATIVE_METHOD
                    else "Divisive"
                ),
            )
        ax.set_title(f"{scenario_name} | {encoding}")
        ax.set_xlabel("Requested number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_xticks(list(K_VALUES))
        ax.legend()
fig.suptitle("Hierarchical clustering silhouette scores (sampled data)")
plt.tight_layout()
score_plot = OUTPUT_DIR / "hierarchical_silhouette.png"
fig.savefig(score_plot, dpi=150, bbox_inches="tight")
print(f"\nSaved silhouette plot: {score_plot}")
plt.show()

#Agglomerative dendrograms compare Original and IQR samples
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (scenario_name, encoding) in zip(
    axes,
    [
        ("Original", "Midpoint"),
        ("IQR excluded", "Midpoint"),
        ("Original", "One-hot"),
        ("IQR excluded", "One-hot"),
    ],
):
    result = hierarchical_results[
        (scenario_name, encoding, AGGLOMERATIVE_METHOD)
    ]
    dendrogram(
        result["tree"],
        truncate_mode="lastp",
        p=30,
        show_leaf_counts=True,
        no_labels=True,
        ax=ax,
    )
    ax.set_title(f"{encoding} | {scenario_name}", fontsize=10, pad=10)
    ax.set_xlabel("Last 30 merges")
    ax.set_ylabel("Merge distance")
fig.suptitle("Ward agglomerative dendrograms (2,000-row samples)", y=1.02)
fig.subplots_adjust(wspace=0.35, top=0.84, bottom=0.15)
dendrogram_plot = OUTPUT_DIR / "hierarchical_dendrograms.png"
fig.savefig(dendrogram_plot, dpi=150, bbox_inches="tight")
print(f"Saved dendrogram plot: {dendrogram_plot}")
plt.show()

#Separate PCA figures compare Original/IQR for Agglomerative and Divisive
for encoding in ("Midpoint", "One-hot"):
    fig, axes = plt.subplots(
        len(scenarios) * len(CLUSTERING_METHODS),
        len(K_VALUES),
        figsize=(18, 12),
    )
    row = 0
    for scenario_name in scenarios:
        for method in CLUSTERING_METHODS:
            result = hierarchical_results[(scenario_name, encoding, method)]
            projection = PCA(n_components=2).fit_transform(result["features"])
            for ax, k in zip(axes[row], K_VALUES):
                labels = result["labels_by_k"][k]
                scatter = ax.scatter(
                    projection[:, 0],
                    projection[:, 1],
                    c=labels,
                    cmap="tab10",
                    s=7,
                    alpha=0.55,
                )
                ax.set_title(f"k={k}", fontsize=9, pad=7)
                ax.set_xlabel("Principal component 1")
                ax.set_ylabel("")
            row += 1

    fig.suptitle(
        f"{encoding} age encoding: hierarchical PCA projections",
        fontsize=14,
        y=0.995,
    )
    row_labels = [
        ("Original", "Agglomerative (Ward)"),
        ("Original", "Divisive"),
        ("IQR excluded", "Agglomerative (Ward)"),
        ("IQR excluded", "Divisive"),
    ]
    for row_index, (scenario_name, method_label) in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(
            f"PC2\n{scenario_name}\n{method_label}",
            fontsize=9,
        )
    fig.subplots_adjust(
        left=0.11,
        right=0.99,
        top=0.94,
        bottom=0.07,
        hspace=0.5,
        wspace=0.3,
    )
    pca_plot = OUTPUT_DIR / f"hierarchical_pca_{encoding.lower().replace('-', '_')}.png"
    fig.savefig(pca_plot, dpi=120)
    print(f"Saved PCA plot: {pca_plot}")
    plt.show()