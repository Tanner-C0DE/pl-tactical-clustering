import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. DATA LOADING
def load_fbref(file_name):
    df = pd.read_csv(file_name, skiprows=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

df_def = load_fbref('def_actions.csv')
df_pass = load_fbref('passing.csv')
df_poss = load_fbref('possession.csv')
df_std = load_fbref('squad_standard.csv')

# 2. MERGE & FEATURE SELECTION
df = df_std[['Squad', 'Poss', 'PrgP', 'PrgC']].merge(
    df_def[['Squad', 'Att 3rd', 'Tkl%', 'Clr', 'Int']], on='Squad'
).merge(
    df_pass[['Squad', 'Cmp%', 'Cmp%.3']], on='Squad'
).merge(
    df_poss[['Squad', 'Att Pen']], on='Squad'
)

feature_map = {
    'Poss': 'Possession %',
    'PrgP': 'Pass Progression',
    'Att 3rd': 'High Pressing',
    'Att Pen': 'Box Presence',
    'Clr': 'Clearances',
    'Int': 'Interceptions',
    'Cmp%.3': 'Long Ball Acc %',
    'Tkl%': 'Tackle Win %'
}

# 3. PREPROCESSING
X = df[list(feature_map.keys())].set_index(df['Squad'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_scaled = StandardScaler().fit_transform(X)

# 4. K-MEANS & MANUAL SORTING
# Initial Cluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
initial_clusters = kmeans.fit_predict(X_scaled)

# SORTING LOGIC: We want highest possession = Group 0
# Calculate mean possession for each initial cluster
cluster_possession = []
for i in range(3):
    mean_poss = X.iloc[initial_clusters == i]['Poss'].mean()
    cluster_possession.append((i, mean_poss))

# Sort clusters by possession descending (Highest Poss first)
# This mapping ensures Dominators = 0, Mid-Block = 1, Low-Block = 2
sorted_clusters = sorted(cluster_possession, key=lambda x: x[1], reverse=True)
cluster_map = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
df['Cluster'] = [cluster_map[c] for c in initial_clusters]

# 5. PCA FOR SCATTER PLOT
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

# --- VISUAL 1: THE TACTICAL MAP (PCA Scatter) ---
plt.figure(figsize=(12, 8))
# Using a custom color palette for consistent group identity
colors = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"} # Green, Blue, Red
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette=colors, s=160, alpha=0.9)
for i in range(df.shape[0]):
    plt.text(df.PCA1[i]+0.06, df.PCA2[i]+0.06, df.Squad[i], fontsize=9)

plt.xlabel('PCA Axis 1 (Control & Progression)')
plt.ylabel('PCA Axis 2 (Defensive Volume & Directness)')
plt.grid(alpha=0.2)
plt.title('Premier League Tactical Map (Sorted by Dominance)', fontsize=16)
plt.savefig('pca_tactical_map_sorted.png')

# --- VISUAL 2: THE TACTICAL SCORECARD (Sorted Heatmap) ---
cluster_means = df.groupby('Cluster')[list(feature_map.keys())].mean()
cluster_means.columns = [feature_map[col] for col in cluster_means.columns]
cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

# Labels for Y-axis including sorted teams
cluster_labels = []
group_names = ["0: Dominators", "1: Mid-Block", "2: Low-Block"]
for i in range(3):
    teams = df[df['Cluster'] == i]['Squad'].tolist()
    label = f"{group_names[i]}\n" + "\n".join([", ".join(teams[j:j+4]) for j in range(0, len(teams), 4)])
    cluster_labels.append(label)

plt.figure(figsize=(16, 10))
sns.heatmap(cluster_norm, annot=cluster_means, fmt=".1f", cmap="RdYlGn", 
            yticklabels=cluster_labels, cbar_kws={'label': 'Relative Strength'})

plt.title('Tactical Identity Scorecard (Groups Sorted by Possession)', fontsize=18, pad=30)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('tactical_scorecard_sorted.png')

print("Sorted successfully! Check 'tactical_scorecard_sorted.png' to see the Dominators on top.")