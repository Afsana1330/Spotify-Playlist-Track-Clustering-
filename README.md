# Spotify-Playlist-Track-Clustering-
An interactive Streamlit web application that groups Spotify tracks and vibes using K-Means machine learning, featuring dynamic Plotly visualizations and the Elbow method


# 🎵 Spotify Track & Vibe Clustering App

An interactive machine learning web application built with **Python** and **Streamlit** that groups songs/tracks into distinct clusters based on their audio features (such as danceability, energy, valence, and acousticness) using **K-Means Unsupervised Learning**.

---

## ✨ Features

* **📁 CSV File Uploader:** Upload your own Spotify dataset or use the built-in sample data generator.
* **🤖 K-Means Clustering:** Dynamically select audio features and choose the number of clusters ($K$) using a slider.
* **📊 Interactive Visualizations:** Powered by **Plotly** for responsive 2D scatter plots and cluster maps.
* **📈 Elbow Method Analysis:** Built-in WCSS line graph to help users visually determine the optimal $K$ value.
* **🎵 Cluster Profiling & Song Lookup:** View average audio metrics per cluster and filter tracks belonging to specific groups.

---

## 🛠️ Tech Stack

* **Python** (Core logic)
* **Streamlit** (Web app framework)
* **Scikit-Learn** (K-Means algorithm & StandardScaler)
* **Pandas & NumPy** (Data manipulation)
* **Plotly** (Interactive charts)

---

