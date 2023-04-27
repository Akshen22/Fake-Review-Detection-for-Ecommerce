# LIBRARIES
import streamlit as st
import pandas as pd
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import base64

# LOAD PICKLE FILES
model = pickle.load(open('stacked_models/stacked-model.pkl', 'rb'))
vectorizer = pickle.load(open('stacked_models/count-vectorizer.pkl', 'rb'))

# FOR STREAMLIT
nltk.download('stopwords')

# TEXT PREPROCESSING
sw = set(stopwords.words('english'))


def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()

    cleaned = []
    stemmed = []

    for token in tokens:
        if token not in sw:
            cleaned.append(token)

    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)


# TEXT CLASSIFICATION
def text_classification(text):
    if len(text) < 1:
        st.write("  ")
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            p = ''.join(str(i) for i in prediction)

            if p == 'True':
                st.success("The review entered is Legitimate.")
            if p == 'False':
                st.error("The review entered is Fraudulent.")


# PAGE FORMATTING AND APPLICATION
def main():
    # sidebar for navigate
    with st.sidebar:
        selected = st.selectbox(
          "Fraud Detection in Online Consumer Reviews Using Machine & Deep Learning Techniques",
          ["Prediction", "About the System & Classifier", "Accuracy"],
        )
    # Prediction Page
    if selected == "Prediction":
        st.title("Fake Review Classifier")
        review = st.text_area("Enter Review: ")
        if st.button("Check"):
            text_classification(review)

    # --CHECKBOXES--
    if selected == "About the System & Classifier":
        st.title("About the System & Classifier")
        st.subheader("Information on the System & Classifier")
        if st.checkbox("About the System"):
            st.markdown(
               "*Fake review detection methods are necessary to find and eliminate fake & misleading reviews from online sites. Consumers often base their purchasing decisions on online reviews, and bogus reviews can skew how people view a good or service, providing false information and possibly hurting sales. Online platforms may identify and eliminate fraudulent reviews by utilising fake review detection models, hence raising the overall calibre and credibility of the reviews that are made available to customers. Customers' trust is maintained and a fair and honest marketplace is supported.In general, fake review detection techniques are crucial tools for preserving the reliability of online platforms and encouraging openness and customer trust.*"
            )

        if st.checkbox("About Classifer"):
            st.markdown('**Model:** Stack: average of 4 algorithms: mnb, svm, lr, mlp')
            st.markdown('**Vectorizer:** Count')
            st.markdown('**Test-Train splitting:** 40% - 60%')
            st.markdown('**Spelling Correction Library:** TextBlob')
            st.markdown('**Stemmer:** PorterStemmer')

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

# RUN MAIN
main()