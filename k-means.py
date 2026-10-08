import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#read the data
df = pd.read_excel("Dataset.xlsx", sheet_name="Synthetic Data")
#creatre the folder for output pics
output_dir = Path(__file__).resolve().parent / "pics"
output_dir.mkdir(exist_ok=True)

#basic data checks
print(f"Dataset shape: {df.shape}")
#check column names
print("\nColumn types:")
print(df.dtypes)
#check missing values and duplicates
print("\nMissing values:")
print(df.isna().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")

#exploratory data analysis
print("\nNumeric summary:")
print(df.describe().T)
#check age group distribution
print("\nAge group counts:")
print(df["AGE"].value_counts(dropna=False).sort_index())

#select binary variables(they are the 'easy' variables to deal with for K-means)
binary_columns = [
    column
    for column in df.select_dtypes(include="number").columns
    if set(df[column].dropna().unique()).issubset({0, 1})
]
#check binary feature proportions
print("\nBinary feature proportions:")
print(df[binary_columns].mean().sort_values(ascending=False))

#check continuous feature skewness and potential outliers
continuous_columns = ["INCOME", "TRB"]
print("\nContinuous feature skewness:")
print(df[continuous_columns].skew())
#check for negative income and zero TRB values (we are not removing them, but worth inspecting)
print(f"\nNegative INCOME rows: {(df['INCOME'] < 0).sum()}")
print(f"Zero TRB rows: {(df['TRB'] == 0).sum()}")

#We dicided to apply IQR rules to see if it helps for scatterplots display
print("\nPotential continuous-feature outliers:")
for column in continuous_columns:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    flagged = (df[column] < lower_bound) | (df[column] > upper_bound)

    print(
        f"{column}: {flagged.sum()} rows ({flagged.mean():.2%}); "
        f"bounds=({lower_bound:.2f}, {upper_bound:.2f})"
    )

#pic for continuous variables' distribution
fig, axes = plt.subplots(1, len(continuous_columns), figsize=(12, 4))
for ax, column in zip(axes, continuous_columns):
    sns.histplot(data=df, x=column, bins=50, ax=ax)
    ax.set_title(f"{column}")
plt.tight_layout()
distribution_plot = output_dir / "INCOME_TRB_distributions.png"
fig.savefig(distribution_plot, dpi=150, bbox_inches="tight")
print(f"Saved distribution plot: {distribution_plot}")
plt.show()

#This is so-called fancy "feature engineering", bc AGE cannot be processed by K-means
#We did both midpoint and one-hot encoding, and see which one works better
age_midpoints = {"18-24": 21, "25-29": 27, "30-34": 32}
age_midpoint = df["AGE"].map(age_midpoints)
if age_midpoint.isna().any():
    unknown_age_groups = df.loc[age_midpoint.isna(), "AGE"].unique().tolist()
    raise ValueError(f"Unmapped AGE categories: {unknown_age_groups}")

X_midpoint = df.drop(columns="AGE").copy()
X_midpoint["AGE"] = age_midpoint

age_one_hot = pd.get_dummies(df["AGE"], prefix="AGE", dtype=int)
X_one_hot = pd.concat([df.drop(columns="AGE"), age_one_hot], axis=1)

#take a look at the age encoding results
print("\nMidpoint-encoded age values:")
print(age_midpoint.value_counts().sort_index())
print("\nOne-hot age columns:")
print(age_one_hot.sum())
print(f"\nX_midpoint shape: {X_midpoint.shape}")
print(f"X_one_hot shape: {X_one_hot.shape}")

#Identify outliers of continuous variables using IQR. Binary variables are not included
outlier_flags = pd.DataFrame(index=df.index)
for column in continuous_columns:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_flags[column] = (df[column] < lower_bound) | (df[column] > upper_bound)

#summary the outlier counts based on IQR
outlier_mask = outlier_flags.any(axis=1)
print("\nIQR outlier exclusion:")
for column in continuous_columns:
    print(f"{column}: {outlier_flags[column].sum()} flagged rows")
print(f"Rows flagged in either continuous variable: {outlier_mask.sum()}")
print(f"Rows retained: {(~outlier_mask).sum()} of {len(df)}")

#We will run K-means on both the original dataset and the IQR-filtered dataset, using both midpoint and one-hot age encodings
data_scenarios = {
    "Original": (df, X_midpoint, X_one_hot),
    "IQR outliers excluded": (
        df.loc[~outlier_mask].copy(),
        X_midpoint.loc[~outlier_mask].copy(),
        X_one_hot.loc[~outlier_mask].copy(),
    ),
}

#For K-means, we will z-score the continuous features (INCOME, TRB, and optionally AGE) while retaining binary indicators as 0/1.
def prepare_features(features, scale_age=False):
   
    columns_to_scale = ["INCOME", "TRB"]
    if scale_age:
        columns_to_scale.append("AGE")

    prepared = features.astype(float).copy()
    scaler = StandardScaler()
    prepared[columns_to_scale] = scaler.fit_transform(prepared[columns_to_scale])
    return prepared

#50k points are too many for scatterplots, so we will sample 5k points for silhouette score and PCA visualization.
k_values = range(2, 7)
random_state = 42
silhouette_sample_size = 5000

#Run K-means for k=2，3，4，5，6, refitting z-score scalers in each scenario.
kmeans_results = {}
for scenario, (_, midpoint_data, one_hot_data) in data_scenarios.items():
    kmeans_results[scenario] = {}
    for encoding, raw_features, scale_age in (
        ("Midpoint", midpoint_data, True),
        ("One-hot", one_hot_data, False),
    ):
        features = prepare_features(raw_features, scale_age=scale_age)
        inertias = []
        silhouette_scores = []
        labels_by_k = {}

        for k in k_values:
            model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = model.fit_predict(features)
            labels_by_k[k] = labels
            inertias.append(model.inertia_)
            score = silhouette_score(
                features,
                labels,
                sample_size=min(silhouette_sample_size, len(features)),
                random_state=random_state,
            )
            silhouette_scores.append(score)

            print(f"\nK-means | {scenario} | {encoding} | k={k}")
            print(f"Silhouette (sample estimate): {score:.4f}")
            print(pd.Series(labels).value_counts().sort_index().to_string())

        kmeans_results[scenario][encoding] = {
            "features": features,
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "labels_by_k": labels_by_k,
        }

#Compare inertia and silhouette for both age encodings before and after IQR filtering.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for row, scenario in enumerate(data_scenarios):
    for encoding in ("Midpoint", "One-hot"):
        results = kmeans_results[scenario][encoding]
        axes[row, 0].plot(
            list(k_values), results["inertias"], marker="o", label=encoding
        )
        axes[row, 1].plot(
            list(k_values),
            results["silhouette_scores"],
            marker="o",
            label=encoding,
        )
    axes[row, 0].set(
        title=f"{scenario}: elbow plot",
        xlabel="Number of clusters (k)",
        ylabel="Inertia",
    )
    axes[row, 1].set(
        title=f"{scenario}: silhouette (sample estimate)",
        xlabel="Number of clusters (k)",
        ylabel="Silhouette score",
    )
    for ax in axes[row]:
        ax.set_xticks(list(k_values))
        ax.legend()
plt.tight_layout()
metrics_plot = output_dir / "kmeans_metrics.png"
fig.savefig(metrics_plot, dpi=150, bbox_inches="tight")
print(f"Saved K-means metrics plot: {metrics_plot}")
plt.show()

#To visualize the clusters, we will use PCA to reduce the features to 2D 
def plot_pca_clusters(results, encoding, scenarios, filename, title):
    fig, axes = plt.subplots(len(scenarios), len(k_values), figsize=(15, 4.5 * len(scenarios)))
    if len(scenarios) == 1:
        axes = np.array([axes])

    for row, scenario in enumerate(scenarios):
        features = results[scenario][encoding]["features"]
        projection = PCA(n_components=2, random_state=random_state).fit_transform(features)
        plot_positions = np.random.default_rng(random_state).choice(
            len(features),
            size=min(5000, len(features)),
            replace=False,
        )
        for ax, k in zip(axes[row], k_values):
            labels = results[scenario][encoding]["labels_by_k"][k]
            scatter = ax.scatter(
                projection[plot_positions, 0],
                projection[plot_positions, 1],
                c=labels[plot_positions],
                cmap="tab10",
                s=8,
                alpha=0.6,
            )
            ax.set_title(f"{scenario}, k={k}")
            ax.set_xlabel("Principal component 1")
            ax.set_ylabel("Principal component 2")

    fig.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved PCA cluster plot: {plot_path}")
    plt.show()

scenario_names = list(data_scenarios)
plot_pca_clusters(
    kmeans_results,
    "Midpoint",
    scenario_names,
    "kmeans_pca_clusters.png",
    "Midpoint K-means in PCA projections",
)
plot_pca_clusters(
    kmeans_results,
    "One-hot",
    scenario_names,
    "kmeans_pca_one_hot.png",
    "One-hot K-means in PCA projections",
)

#For robustness, we perform it again on log-transformed INCOME and TRB values, after excluding IQR outliers and non-positives
log_df = df.loc[(~outlier_mask) & (df["INCOME"] > 0)].copy()
print(
    "\nLog-transformed analysis rows: "
    f"{len(log_df)} (excluded {((~outlier_mask) & (df['INCOME'] <= 0)).sum()} "
    "non-positive INCOME rows after IQR filtering)"
)

log_midpoint = X_midpoint.loc[log_df.index].copy()
log_one_hot = X_one_hot.loc[log_df.index].copy()
for features in (log_midpoint, log_one_hot):
    features["INCOME"] = np.log(features["INCOME"])
    features["TRB"] = np.log1p(features["TRB"])

log_feature_sets = {
    "Midpoint": prepare_features(log_midpoint, scale_age=True),
    "One-hot": prepare_features(log_one_hot),
}
log_kmeans_results = {}
for encoding, features in log_feature_sets.items():
    inertias = []
    scores = []
    labels_by_k = {}

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(features)
        labels_by_k[k] = labels
        inertias.append(model.inertia_)
        score = silhouette_score(
            features,
            labels,
            sample_size=min(silhouette_sample_size, len(features)),
            random_state=random_state,
        )
        scores.append(score)
        print(f"\nK-means | IQR excluded + log | {encoding} | k={k}")
        print(f"Silhouette (sample estimate): {score:.4f}")
        print(pd.Series(labels).value_counts().sort_index().to_string())

    log_kmeans_results[encoding] = {
        "features": features,
        "inertias": inertias,
        "silhouette_scores": scores,
        "labels_by_k": labels_by_k,
    }

#Also draw plots for the IQR-filtered, log-transformed K-means
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for encoding, results in log_kmeans_results.items():
    axes[0].plot(list(k_values), results["inertias"], marker="o", label=encoding)
    axes[1].plot(
        list(k_values),
        results["silhouette_scores"],
        marker="o",
        label=encoding,
    )
axes[0].set(title="Log-transformed data: elbow plot", xlabel="k", ylabel="Inertia")
axes[1].set(
    title="Log-transformed data: silhouette",
    xlabel="k",
    ylabel="Silhouette score",
)
for ax in axes:
    ax.set_xticks(list(k_values))
    ax.legend()
plt.tight_layout()
log_metrics_plot = output_dir / "kmeans_metrics_log_iqr.png"
fig.savefig(log_metrics_plot, dpi=150, bbox_inches="tight")
print(f"Saved log-analysis metrics plot: {log_metrics_plot}")
plt.show()

log_results_for_plot = {"IQR excluded + log": log_kmeans_results}
for encoding, filename in (
    ("Midpoint", "kmeans_pca_midpoint_log_iqr.png"),
    ("One-hot", "kmeans_pca_one_hot_log_iqr.png"),
):
    plot_pca_clusters(
        log_results_for_plot,
        encoding,
        ["IQR excluded + log"],
        filename,
        f"{encoding} K-means after IQR exclusion and log transformation",
    )

#We would like to look at the profiles for clusters(IQR applied, z-score scaled, both midpoint and one-hot encoding for AGE)
profile_scenario = "IQR outliers excluded"
profile_result = kmeans_results[profile_scenario]["Midpoint"]
profile_source = data_scenarios[profile_scenario][0].loc[
    profile_result["features"].index
].copy()
overall_binary_rates = profile_source[binary_columns].mean()
profile_summary_rows = []
binary_driver_rows = []

for k, labels in profile_result["labels_by_k"].items():
    profile_data = profile_source.copy()
    profile_data["cluster"] = labels + 1
    group_sizes = profile_data["cluster"].value_counts().sort_index()
    #numerica summary for INCOME and TRB
    numeric_summary = profile_data.groupby("cluster").agg(
        INCOME_mean=("INCOME", "mean"),
        INCOME_median=("INCOME", "median"),
        TRB_mean=("TRB", "mean"),
        TRB_median=("TRB", "median"),
    )
    age_shares = pd.crosstab(
        profile_data["cluster"],
        profile_data["AGE"],
        normalize="index",
    ).reindex(columns=sorted(profile_source["AGE"].unique()), fill_value=0)
    age_shares.columns = [f"AGE_{age}_share" for age in age_shares.columns]
    binary_rates = profile_data.groupby("cluster")[binary_columns].mean()
    binary_rates.columns = [f"{column}_rate" for column in binary_rates.columns]

    group_summary = pd.concat(
        [
            group_sizes.rename("cluster_size"),
            (group_sizes / len(profile_data)).rename("cluster_share"),
            numeric_summary,
            age_shares,
            binary_rates,
        ],
        axis=1,
    ).reset_index()
    group_summary.insert(0, "k", k)
    profile_summary_rows.append(group_summary)

    for cluster_id, cluster_rows in profile_data.groupby("cluster"):
        cluster_rates = cluster_rows[binary_columns].mean()
        differences = cluster_rates - overall_binary_rates
        for feature in binary_columns:
            baseline_rate = overall_binary_rates[feature]
            cluster_rate = cluster_rates[feature]
            binary_driver_rows.append(
                {
                    "k": k,
                    "cluster": cluster_id,
                    "feature": feature,
                    "cluster_rate": cluster_rate,
                    "overall_rate": baseline_rate,
                    "difference_percentage_points": 100
                    * differences[feature],
                    "lift_vs_overall": (
                        cluster_rate / baseline_rate
                        if baseline_rate > 0
                        else np.nan
                    ),
                }
            )

    print(f"\nCluster profiles | IQR excluded | midpoint | k={k}")
    print(
        group_summary[
            [
                "cluster",
                "cluster_size",
                "cluster_share",
                "INCOME_median",
                "TRB_median",
                *age_shares.columns,
            ]
        ].to_string(index=False, formatters={"cluster_share": "{:.1%}".format})
    )
    driver_table = pd.DataFrame(
        [
            row
            for row in binary_driver_rows
            if row["k"] == k
        ]
    )
    for cluster_id in sorted(profile_data["cluster"].unique()):
        top_drivers = (
            driver_table[driver_table["cluster"] == cluster_id]
            .assign(
                absolute_difference=lambda table: table[
                    "difference_percentage_points"
                ].abs()
            )
            .nlargest(3, "absolute_difference")
        )
        descriptions = [
            (
                f"{row.feature}: {row.cluster_rate:.1%} "
                f"(overall {row.overall_rate:.1%}, "
                f"{row.difference_percentage_points:+.1f} pp)"
            )
            for row in top_drivers.itertuples()
        ]
        print(f"Cluster {cluster_id} most distinctive binary features:")
        for description in descriptions:
            print(f"  - {description}")

profile_summary = pd.concat(profile_summary_rows, ignore_index=True)
binary_driver_summary = pd.DataFrame(binary_driver_rows)
profile_path = output_dir / "kmeans_midpoint_iqr_cluster_profiles.csv"
drivers_path = output_dir / "kmeans_midpoint_iqr_binary_drivers.csv"
profile_summary.to_csv(profile_path, index=False, encoding="utf-8-sig")
binary_driver_summary.to_csv(drivers_path, index=False, encoding="utf-8-sig")
print(f"\nSaved cluster profiles: {profile_path}")
print(f"Saved binary feature contrasts: {drivers_path}")

#For one-hot encoding
one_hot_result = kmeans_results[profile_scenario]["One-hot"]
one_hot_profile_source = data_scenarios[profile_scenario][0].loc[
    one_hot_result["features"].index
].copy()
one_hot_overall_binary_rates = one_hot_profile_source[binary_columns].mean()
one_hot_profile_rows = []
one_hot_driver_rows = []

for k, labels in one_hot_result["labels_by_k"].items():
    profile_data = one_hot_profile_source.copy()
    profile_data["cluster"] = labels + 1
    group_sizes = profile_data["cluster"].value_counts().sort_index()

    numeric_summary = profile_data.groupby("cluster").agg(
        INCOME_mean=("INCOME", "mean"),
        INCOME_median=("INCOME", "median"),
        TRB_mean=("TRB", "mean"),
        TRB_median=("TRB", "median"),
    )
    age_shares = pd.crosstab(
        profile_data["cluster"],
        profile_data["AGE"],
        normalize="index",
    ).reindex(columns=sorted(one_hot_profile_source["AGE"].unique()), fill_value=0)
    age_shares.columns = [f"AGE_{age}_share" for age in age_shares.columns]
    binary_rates = profile_data.groupby("cluster")[binary_columns].mean()
    binary_rates.columns = [f"{column}_rate" for column in binary_rates.columns]

    group_summary = pd.concat(
        [
            group_sizes.rename("cluster_size"),
            (group_sizes / len(profile_data)).rename("cluster_share"),
            numeric_summary,
            age_shares,
            binary_rates,
        ],
        axis=1,
    ).reset_index()
    group_summary.insert(0, "k", k)
    one_hot_profile_rows.append(group_summary)

    for cluster_id, cluster_rows in profile_data.groupby("cluster"):
        cluster_rates = cluster_rows[binary_columns].mean()
        differences = cluster_rates - one_hot_overall_binary_rates
        for feature in binary_columns:
            baseline_rate = one_hot_overall_binary_rates[feature]
            cluster_rate = cluster_rates[feature]
            one_hot_driver_rows.append(
                {
                    "k": k,
                    "cluster": cluster_id,
                    "feature": feature,
                    "cluster_rate": cluster_rate,
                    "overall_rate": baseline_rate,
                    "difference_percentage_points": 100 * differences[feature],
                    "lift_vs_overall": (
                        cluster_rate / baseline_rate
                        if baseline_rate > 0
                        else np.nan
                    ),
                }
            )

    print(f"\nCluster profiles | IQR excluded | one-hot | k={k}")
    print(
        group_summary[
            [
                "cluster",
                "cluster_size",
                "cluster_share",
                "INCOME_median",
                "TRB_median",
                *age_shares.columns,
            ]
        ].to_string(index=False, formatters={"cluster_share": "{:.1%}".format})
    )

one_hot_profile_summary = pd.concat(one_hot_profile_rows, ignore_index=True)
one_hot_binary_driver_summary = pd.DataFrame(one_hot_driver_rows)
one_hot_profile_path = output_dir / "kmeans_one_hot_iqr_cluster_profiles.csv"
one_hot_drivers_path = output_dir / "kmeans_one_hot_iqr_binary_drivers.csv"
one_hot_profile_summary.to_csv(
    one_hot_profile_path, index=False, encoding="utf-8-sig"
)
one_hot_binary_driver_summary.to_csv(
    one_hot_drivers_path, index=False, encoding="utf-8-sig"
)
print(f"\nSaved one-hot cluster profiles: {one_hot_profile_path}")
print(f"Saved one-hot binary feature contrasts: {one_hot_drivers_path}")
