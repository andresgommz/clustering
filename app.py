import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, clustering, evaluate

st.title('Clustering App')

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

features = st.multiselect('Select the features for clustering', df.columns.to_list())

if len(features) < 2:
    st.warning('Select at least 2 features')
    st.stop()

n_clusters = st.slider('Number of clusters', 2, 10, 3)
linkage = st.selectbox('Select linkage', ['ward', 'single', 'complete'])

X = preprocessing(df, features)
labels = clustering(X, n_clusters=n_clusters, linkage=linkage)
df['cluster'] = labels
st.subheader('Clustered dataset')
st.dataframe(df)
silhouette, davies = evaluate(X, labels)
st.write('Silhouette', silhouette)
st.write('Davies Index', davies)
if len(features) == 2:
    fig, ax = plt.subplots()
    ax.scatter(df[features[0]], df[features[1]],c=labels)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    st.pyplot(fig)
else:
    st.info('Select exactly 2 features for visualization.')