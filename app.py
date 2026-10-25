import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Spotify Track Clustering App",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Track & Vibe Clustering App")
st.markdown("Group songs into clusters based on their audio features using **K-Means Machine Learning**.")

# --- DATA LOADING FUNCTION ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Generate a realistic mock Spotify dataset if no file is uploaded
        np.random.seed(42)
        n_samples = 300
        df = pd.DataFrame({
            'track_name': [f"Song {i+1}" for i in range(n_samples)],
            'artist_name': np.random.choice(["Artist A", "Artist B", "Artist C", "Artist D", "Artist E"], n_samples),
            'danceability': np.random.uniform(0.1, 0.9, n_samples),
            'energy': np.random.uniform(0.1, 0.95, n_samples),
            'valence': np.random.uniform(0.0, 1.0, n_samples),  # Positivity/Mood
            'acousticness': np.random.uniform(0.0, 0.99, n_samples),
            'tempo': np.random.uniform(60, 180, n_samples)
        })
    return df

# Sidebar for controls
st.sidebar.header("1. Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Spotify CSV (Optional)", type=["csv"])

df = load_data(uploaded_file)

if uploaded_file is None:
    st.sidebar.info("💡 Using built-in sample data. Upload your own CSV anytime!")

# Show raw data preview checkbox
if st.sidebar.checkbox("Preview Raw Dataset"):
    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head())

# Select audio features for clustering
audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
# Filter features to ensure they exist in the dataframe
available_features = [f for f in audio_features if f in df.columns]

st.sidebar.header("2. K-Means Parameters")
selected_features = st.sidebar.multiselect(
    "Select Features for Clustering",
    options=available_features,
    default=['danceability', 'energy', 'valence']
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features from the sidebar to perform clustering.")
    st.stop()

# Choose K value
k_clusters = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=8, value=4)

# --- MACHINE LEARNING PROCESSING ---
# 1. Scale features (Critical for K-Means so tempo doesn't overpower danceability)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_features])

# 2. Elbow Method calculation for visual inspection
wcss = []
k_range = range(1, 10)
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(scaled_data)
    wcss.append(kmeans_test.inertia_)

# 3. Fit K-Means with user-selected K
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)
df['Cluster'] = df['Cluster'].astype(str) # Convert to string for discrete color mapping

# --- LAYOUT: TABS FOR VISUALIZATION ---
tab1, tab2, tab3 = st.tabs(["📊 Cluster Visualization", "📈 Elbow Analysis", "🎵 Explore Clusters"])

with tab1:
    st.subheader(f"Interactive 2D Cluster Map (K = {k_clusters})")
    
    if len(selected_features) >= 2:
        x_axis = selected_features[0]
        y_axis = selected_features[1]
        
        # Plotly scatter plot
        fig = px.scatter(
            df, 
            x=x_axis, 
            y=y_axis, 
            color='Cluster',
            hover_data=['track_name', 'artist_name'] if 'track_name' in df.columns else None,
            title=f"Clusters based on {x_axis} vs {y_axis}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Select at least 2 features to render the scatter plot.")

with tab2:
    st.subheader("The Elbow Method (Optimal K Finder)")
    st.markdown("Look for the 'elbow' bend point where the curve flattens out—that represents the ideal number of clusters.")
    
    elbow_fig = px.line(
        x=list(k_range), 
        y=wcss, 
        markers=True,
        labels={'x': 'Number of Clusters (K)', 'y': 'WCSS (Within-Cluster Sum of Squares)'},
        title="Elbow Graph",
        template="plotly_dark"
    )
    st.plotly_chart(elbow_fig, use_container_width=True)

with tab3:
    st.subheader("Cluster Profile & Song Lookup")
    
    # Calculate average feature values per cluster
    cluster_profile = df.groupby('Cluster')[selected_features].mean().reset_index()
    st.markdown("**Average Audio Profile per Cluster:**")
    st.dataframe(cluster_profile.style.background_gradient(cmap="Blues"))
    
    # Filter songs by cluster
    selected_cluster_lookup = st.selectbox("Select a Cluster to view tracks:", options=df['Cluster'].unique())
    filtered_songs = df[df['Cluster'] == selected_cluster_lookup]
    
    st.markdown(f"Songs belonging to **Cluster {selected_cluster_lookup}** ({len(filtered_songs)} tracks found):")
    st.dataframe(filtered_songs[['track_name', 'artist_name'] + selected_features] if 'track_name' in df.columns else filtered_songs)
