import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import plotly.express as px
import re

# Load the LaBSE model
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/LaBSE")

model = load_model()

def fetch_sitemap_urls(domain):
    """Fetch and parse URLs from sitemaps, excluding images and handling nested sitemaps."""
    domain = domain.replace("https://", "").replace("http://", "").strip("/")
    sitemap_urls = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://{domain}/robots.txt"
    ]
    all_urls = []

    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url, headers={"User-Agent": "SiteFocusTool/1.0"}, timeout=10)
            response.raise_for_status()
            if "robots.txt" in sitemap_url:
                for line in response.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        nested_sitemap_url = line.split(":", 1)[1].strip()
                        all_urls.extend(fetch_sitemap_urls_from_xml(nested_sitemap_url, domain, recursive=True))
            else:
                all_urls.extend(fetch_sitemap_urls_from_xml(sitemap_url, domain, recursive=True))
        except requests.RequestException:
            continue
    return list(set(all_urls))

def fetch_sitemap_urls_from_xml(sitemap_url, domain, recursive=False):
    """Fetch URLs from a sitemap XML file."""
    urls = []
    try:
        response = requests.get(sitemap_url, headers={"User-Agent": "SiteFocusTool/1.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml-xml")  # Use lxml parser
        if soup.find_all("sitemap"):
            for sitemap in soup.find_all("sitemap"):
                loc = sitemap.find("loc").text
                if recursive:
                    urls.extend(fetch_sitemap_urls_from_xml(loc, domain, recursive=True))
        else:
            for loc in soup.find_all("loc"):
                url = loc.text
                if not re.search(r"\.(jpg|jpeg|png|gif|svg|webp|bmp|tif|tiff)$", url, re.IGNORECASE):
                    urls.append(url)
    except requests.RequestException:
        pass
    return urls

def clean_text_from_url(url, domain):
    """Clean URL by removing root domain and extracting readable text."""
    domain = domain.replace("https://", "").replace("http://", "").strip("/")
    url = url.replace(f"https://{domain}/", "").replace(f"http://{domain}/", "")
    text = re.sub(r"[^\w\s]", " ", url)
    text = text.replace("/", " ").replace("_", " ").replace("-", " ")
    return text.strip()

def compute_embeddings(data):
    """Generate normalized embeddings for the cleaned text."""
    data["Embedding"] = data["Cleaned Text"].apply(lambda text: model.encode(text))
    data["Embedding"] = data["Embedding"].apply(lambda emb: emb / norm(emb))  # Normalize
    return data

def calculate_site_focus_and_radius(embeddings):
    """Calculate site focus score and site radius."""
    centroid_embedding = np.mean(embeddings, axis=0)
    deviations = [1 - cosine_similarity([embedding], [centroid_embedding])[0][0] for embedding in embeddings]
    site_radius = np.mean(deviations)
    site_focus_score = max(0, 1 - site_radius)
    return site_focus_score, site_radius, centroid_embedding, deviations

def plot_gradient_strip_with_indicator(score, title):
    """Visualize the score as a gradient strip with an indicator."""
    plt.figure(figsize=(8, 1))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    plt.imshow(gradient, aspect="auto", cmap="RdYlGn_r")  # Red to Green reversed for correct mapping
    plt.axvline(x=score * 256, color="black", linestyle="--", linewidth=2)
    plt.gca().set_axis_off()
    plt.title(f"{title}: {score * 100:.2f}%")
    plt.show()
    st.pyplot(plt)

def plot_3d_tsne(embeddings, urls, centroid, deviations):
    """Interactive 3D t-SNE scatter plot with hover labels."""
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
    tsne_results = tsne.fit_transform(np.vstack([embeddings, centroid]))
    centroid_tsne = tsne_results[-1]  # Last point is the centroid
    tsne_results = tsne_results[:-1]  # Remaining points are pages

    fig = px.scatter_3d(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        z=tsne_results[:, 2],
        color=deviations,
        color_continuous_scale="RdYlGn_r",
        hover_name=urls,
        labels={"color": "Deviation"},
        title="3D t-SNE Projection of Page Embeddings"
    )
    fig.add_scatter3d(
        x=[centroid_tsne[0]],
        y=[centroid_tsne[1]],
        z=[centroid_tsne[2]],
        mode="markers",
        marker=dict(size=15, color="green"),
        name="Centroid"
    )
    st.plotly_chart(fig)

def plot_spherical_distances_optimized(deviations, embeddings, urls):
    """Improved scatter plot showing distances in a spherical layout with better angle distribution."""
    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    num_points = len(deviations)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # Spread angles evenly

    # Create polar scatter plot
    fig = px.scatter_polar(
        r=deviations,
        theta=np.degrees(angles),
        color=deviations,
        color_continuous_scale="RdYlGn_r",
        title="Optimized Spherical Plot of Page Distances from Centroid",
        labels={"color": "Deviation"}
    )
    # Update traces to show text (labels) only on hover
    fig.update_traces(
        mode="markers",  # Display only markers by default
        hovertemplate="%{text}<extra></extra>",  # Show text on hover
        text=urls  # Set URLs as hover labels
    )
    st.plotly_chart(fig)

# Streamlit Interface
st.title("SiteFocus Tool")

domain = st.text_input("Enter domain:", placeholder="example.com")

if st.button("START"):
    if domain:
        urls = fetch_sitemap_urls(domain)
        if not urls:
            st.error("No URLs found. Please check the domain and try again.")
        else:
            cleaned_texts = [clean_text_from_url(url, domain) for url in urls]
            embeddings = np.array([model.encode(text) / norm(model.encode(text)) for text in cleaned_texts])
            site_focus_score, site_radius, centroid, deviations = calculate_site_focus_and_radius(embeddings)

            # Visualize siteFocusScore
            st.subheader("siteFocusScore")
            st.markdown("**Description:** The siteFocusScore reflects how tightly aligned a site's content is to a single thematic area. A higher score indicates greater thematic focus, which can improve topical authority in SEO.")
            plot_gradient_strip_with_indicator(site_focus_score, "siteFocusScore")

            # Visualize siteRadius
            st.subheader("siteRadius")
            st.markdown("**Description:** The siteRadius measures how far individual pages deviate from the site's central theme. A smaller radius indicates higher consistency across the site, which is beneficial for SEO.")
            plot_gradient_strip_with_indicator(site_radius, "siteRadius")

            # Sorted dataframe by closeness to centroid
            st.subheader("Pages Closest to Centroid")
            distances = [1 - dev for dev in deviations]
            df = pd.DataFrame({"URL": urls, "Proximity to Centroid": distances})
            df_sorted = df.sort_values(by="Proximity to Centroid", ascending=False)
            st.dataframe(df_sorted)

            # Interactive 3D t-SNE plot
            st.subheader("3D t-SNE Projection")
            plot_3d_tsne(embeddings, urls, centroid, deviations)

            # Optimized spherical distance plot
            st.subheader("Spherical Distance Plot")
            plot_spherical_distances_optimized(deviations, embeddings, urls)
Hyper Icon
