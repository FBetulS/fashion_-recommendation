import streamlit as st
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("Fashion Recommendation System")

@st.cache_resource
def load_data():
    features = pickle.load(open('Images_features.pkl', 'rb'))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features)
    return features, filenames, neighbors

@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return tf.keras.Sequential([model, GlobalMaxPool2D()])

def extract_features(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))
    result = model.predict(img_preprocessed).flatten()
    return result / norm(result)

features, filenames, neighbors = load_data()
model = load_model()

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])
num_recs = st.slider("Number of recommendations", 1, 10, 5)

if uploaded_file:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Your Image")
        img = Image.open(uploaded_file)
        st.image(img, width=250)
    
    with col2:
        st.subheader("Similar Items")
        features_uploaded = extract_features(img, model)
        distances, indices = neighbors.kneighbors([features_uploaded])
        
        cols = st.columns(min(5, num_recs))
        shown = 0
        
        for i in range(len(indices[0])):
            if shown >= num_recs:
                break
            try:
                path = os.path.join('images', os.path.basename(filenames[indices[0][i]]))
                with cols[shown % 5]:
                    st.image(Image.open(path), width=150)
                    st.write(f"Similarity: {1 - distances[0][i]:.2f}")
                shown += 1
            except:
                continue

st.markdown("---")
st.write("Fashion Recommendation Engine")