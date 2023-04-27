import pickle
import streamlit as st
import pandas as pd
from utils import *
import base64


# sidebar for navigate
with st.sidebar:
    selected = st.selectbox(
        "Fraud Detection in Online Consumer Reviews Using GNN Technique",
        ["Prediction", "About the System", "Accuracy"],
    )

# Prediction Page
if selected == "Prediction":
    # page title
    st.title("Fake Review Classifier")
    st.subheader("Enter Review : ")
    tweet_data = st.text_area("", value="", height=100)
    select_notebook_file = st.selectbox(
        "Select the model versions to run over",
        [
            "gnn_model",
        ],
    )

    if select_notebook_file == "gnn_model":
        select_model = st.selectbox(
            "Select model",
            ["gnn"],
        )
        if select_model == "gnn":
            gnn_model_tokenizer_path = "gnn_tokenizer.pkl"
            gnn_model_path = "gnn_model/cached_pmi_model.p"

    submit_button = st.button("Check")

    if submit_button and select_notebook_file == "gnn_model":
        if select_model == "gnn":
            prediction = predict_classes(
                tweet_data, "gnn_tokenizer.pkl", "gnn_model/cached_pmi_model.p"
            )
            label_arr = ["Legitimate", "Fraudulent"]
            st.success("checking done", icon="✅")
            st.title(f"We found this review to be {label_arr[prediction]}")
        else:
            prediction = predict_classes(tweet_data, gnn_model_path, gnn_model_tokenizer_path)
            st.success("checking done", icon="✅")
            st.title("We found this review to be " + str(prediction))

    elif submit_button and tweet_data:
        pred = make_pred(tweet_data, select_model, select_notebook_file)
        st.success("checking done!", icon="✅")
        if pred[0] == 0 or pred[0] == 1:
            label_arr = ["Legitimate", "Fraudulent"]
            prediction = label_arr[pred[0]]
        else:
            prediction = pred[0]
        st.title("We found this review to be " + str(prediction))

if selected == "About the System":
    # page title
    st.title("About the System")

    st.markdown(
        "*Fake review detection methods are necessary to find and eliminate fake & misleading reviews from online sites. Consumers often base their purchasing decisions on online reviews, and bogus reviews can skew how people view a good or service, providing false information and possibly hurting sales. Online platforms may identify and eliminate fraudulent reviews by utilising fake review detection models, hence raising the overall calibre and credibility of the reviews that are made available to customers. Customers' trust is maintained and a fair and honest marketplace is supported.In general, fake review detection techniques are crucial tools for preserving the reliability of online platforms and encouraging openness and customer trust.*"
    )

if selected == "Accuracy":
    st.title("Accuracy")
    st.subheader("Information on the Accuracy via Bar Graph ")
    data = {"name": ["MNB", "SVM", "LR", "MLP Classifier", "Stacked", "LSTM", "GCN"], "Accuracy": [80.4, 84.23, 84.96, 82.91, 86.27, 67.48, 99.7]}
    data = pd.DataFrame(data)
    data = data.set_index("name")
    st.bar_chart(data)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('concept-gift-gift-box-brown-background_185193-88995.avif')