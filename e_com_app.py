import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# page configuration
st.set_page_config(page_title="E-Commerece Systeam",layout="wide")


# load data
@st.cache_data
def load_data():
    df = pd.read_pickle("processed_df.pkl")
    with open("x_processed.pkl", "rb") as f:
        x_processed = pickle.load(f)
    return df, x_processed


df, x_processed = load_data()
#sidebar navigation
 
st.sidebar.title("Menu")

page =st.sidebar.radio("Select page",["Data Analysis","Recommendation System"])

#--------------- Page 1: Data Analysis ------------
if page=="Data Analysis":
    st.title('Data Analysis Dashboard')

    st.write("Jupyter Notebook Analysis")

    brand_pivot = (df.groupby(['brand','cluster']).size()
                   .reset_index(name='Count')
                   .pivot(index='brand',columns='cluster',values='Count'))

    st.write( brand_pivot)

    fig = px.imshow(
    brand_pivot,
    text_auto=True,
    color_continuous_scale="Greens",
    labels=dict(x="cluster",
                y='Brand',
                color='Count'),
    title="Brand Distribution across Cluster",
   )
    st.plotly_chart(fig,use_container_width=True)




  
    model_pivot = (df.groupby(['model','cluster']).size()
                   .reset_index(name='Count')
                   .pivot(index='model',columns='cluster',values='Count'))

    st.write(model_pivot)

    fig = px.bar(
    model_pivot,
    barmode="group",
    title="Model Counts by Cluster ",
    labels={"value":"Count","model":"Model","varible":"Cluster"},
   )
    st.plotly_chart(fig,use_container_width=True)




    cluster_means = df.groupby('cluster')[['performance_rating','camera_rating','helpful_votes','design_rating']].mean().reset_index()

    st.write(cluster_means)

    fig = px.bar(
    cluster_means,
    x='cluster',
    y=['performance_rating','camera_rating','design_rating','helpful_votes'],
    barmode='group',
    title='Cluster Wise Rating Comparsion'
   )
    st.plotly_chart(fig,use_container_width=True)



    negative_df = df[df['design_rating']<2.5]
    st.write(negative_df)

    fig = px.box(
    df,
    x='cluster',
    y='design_rating',
    color='cluster',
    points='all',
    title='Desing Rating Distribution & Low Rating Outliers'
  )
    st.plotly_chart(fig,use_container_width=True)



    fig = px.violin(
    df,
    x='cluster',
    y='battery_life_rating',
    color='cluster',
    box=True,
    points='all',
    title='Rating Density (Look for Bettry Capacity)'
 )
    st.plotly_chart(fig,use_container_width=True)

 

    source_clusters = pd.crosstab(df["source"],df["cluster"])
    st.write(source_clusters)
    fig = px.bar(
    source_clusters,
    text_auto=True,
        labels=dict(x="cluster",
                    y="source",
                    color='Count'),
        title="Source Distribution across Cluster",
    )
    st.plotly_chart(fig,use_container_width=True)

#================page 2: Recommendation syteam==============#

elif page=="Recommendation System":
    st.title("🤖 Advanced Product Recommendation System")
    st.write(
        "Select a reference mobile and apply filters to instantly view tailored recommendations below."
    )

    # Recommendation Function with Dynamic Filtering
    def recommend_mobiles(
        selected_index,
        top_n=4,
        sel_brands=None,
        min_rating=0.0,
        max_price=None,
    ):
        if sel_brands is None:
            sel_brands = []

        target_cluster = df.loc[selected_index, "cluster"]

        # Filter dataset by cluster
        cluster_df = df[df["cluster"] == target_cluster].copy()

        # Apply Brand Filter
        if sel_brands:
            cluster_df = cluster_df[cluster_df["brand"].isin(sel_brands)]

        # Apply Rating Filter
        if "rating" in cluster_df.columns:
            cluster_df = cluster_df[cluster_df["rating"] >= min_rating]

        # Apply Price Filter
        if max_price is not None and "price_usd" in cluster_df.columns:
            cluster_df = cluster_df[cluster_df["price_usd"] <= max_price]

        if cluster_df.empty:
            return pd.DataFrame(), []

        cluster_indices = cluster_df.index
        target_vector = x_processed[selected_index].reshape(1, -1)
        cluster_vectors = x_processed[cluster_indices]

        sim_scores = cosine_similarity(target_vector, cluster_vectors)[0]
        top_cluster_indices = np.argsort(sim_scores)[::-1]

        filtered_indices = []
        filtered_scores = []
        for idx, score in zip(top_cluster_indices, sim_scores[top_cluster_indices]):
            real_idx = cluster_indices[idx]
            if real_idx != selected_index:
                filtered_indices.append(real_idx)
                filtered_scores.append(score)
            if len(filtered_indices) == top_n:
                break

        return df.loc[filtered_indices], filtered_scores

    # UI Controls
    model_col = "model" if "model" in df.columns else df.columns[0]

    selected_phone = st.selectbox(
        "📱 Select Reference Mobile:", df[model_col].unique()
    )

    st.subheader("🔍 Filter Recommendations")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        available_brands = (
            list(df["brand"].dropna().unique()) if "brand" in df.columns else []
        )
        sel_brands = st.multiselect("🏷️ Brand:", available_brands)

    with col2:
        min_rating = st.slider(
            "⭐ Min Rating:", min_value=0.0, max_value=5.0, value=0.0, step=0.5
        )

    with col3:
        if "price_usd" in df.columns:
            max_p_val = (df["price_usd"].max())
            min_p_val = float(df["price_usd"].min())
            max_price = st.slider(
                "💰 Max Price (₹):",
                min_value=min_p_val,
                max_value=max_p_val,
                value=max_p_val,
            )
        
    with col4:
        top_n = st.slider(
            "Top N Recommendations:", min_value=1, max_value=10, value=4
        )

    # Auto-run Recommendations
    selected_idx = df[df[model_col] == selected_phone].index[0]
    recommendations, scores = recommend_mobiles(
        selected_idx,
        top_n=top_n,
        sel_brands=sel_brands,
        min_rating=min_rating,
        max_price=max_price,
    )

    st.write("---")
    st.subheader(f"✨ Recommendations for '{selected_phone}':")

    # Display Results in Vertical Cards
    if recommendations.empty:
        st.warning(
            "⚠️ No products match your selected filters. Please adjust the filter values and try again."
        )
    else:
        for (_, row), score in zip(recommendations.iterrows(), scores):
            with st.container(border=True):
                c_info, c_match = st.columns([3, 1])

                with c_info:
                    st.markdown(f"### 📱 **{row.get('model', 'Mobile')}**")
                    st.write(f"🏷️ **Brand:** {row.get('brand', 'N/A')}")

                    if "price_usd" in row and not pd.isna(row["price_usd"]):
                        st.write(f"💰 **Price_usd:** ₹{row['price_usd']}")

                    if "rating" in row and not pd.isna(row["rating"]):
                        st.write(f"⭐ **Rating:** {row['rating']} / 5")

                    if "country"in row and not pd.isna(row["country"]):
                         st.write(f" 🌐 **Country:** {row['country']} ")
                        
                    if "source" in row and not pd.isna(row["source"]):
                        st.write(f"🌐 **Source:** {row['source']}")

                with c_match:
                    st.success(f"Match Score\n\n### {score*100:.1f}%")



