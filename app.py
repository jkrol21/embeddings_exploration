import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deep_translator import GoogleTranslator
from openai import OpenAI
from sklearn.decomposition import PCA

MODEL_NAME = "text-embedding-3-small"
COLS_PER_ROW = 3
INITIAL_WORDS_EXAMPLE = "king, queen, man, woman"


@st.cache_resource
def get_client():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    return OpenAI()


@st.cache_data
def load_example_vectors():
    example_vectors_df = pd.read_csv("data/example_vectors.csv")
    return example_vectors_df


def get_embedding_from_api(word: str) -> np.array:
    """ """

    word_embedding = (
        client.embeddings.create(input=word, model=MODEL_NAME, encoding_format="float")
        .data[0]
        .embedding
    )

    word_embedding_df = pd.DataFrame(word_embedding).T
    word_embedding_df.columns = [
        "d" + str(i) for i in range(1, word_embedding_df.shape[1] + 1)
    ]
    word_embedding_df["word"] = word

    embedding_vector = word_embedding_df.drop(columns=["word"]).to_numpy()
    embedding_vector = embedding_vector.reshape(-1)

    return embedding_vector


def get_word_embedding(word) -> np.array:
    """ """

    if word in example_vectors_df["word"].values:
        return (
            example_vectors_df[example_vectors_df["word"] == word]
            .drop(columns=["word", "language"])
            .to_numpy()
            .reshape(-1)
        )

    word_embedding = get_embedding_from_api(word)

    return word_embedding


def translate(word, target_language) -> str:
    return GoogleTranslator(source="en", target=target_language).translate(word)


def plot_pca_embeddings(words, vectors, color_mapping, title):
    pca = PCA(n_components=2)
    vectors = np.vstack(vectors)
    principal_components = pca.fit_transform(vectors)
    principal_components = pd.DataFrame(principal_components)
    principal_components.columns = ["PC_1", "PC_2"]
    principal_components["word"] = words

    fig = px.scatter(
        principal_components,
        x="PC_1",
        y="PC_2",
        text=words,
        hover_data={"word": True, "PC_1": False, "PC_2": False},
        color=color_mapping,
        title=title,
    )

    fig.update_traces(
        showlegend=False,
        mode="markers+text",
        marker=dict(
            size=72,
            opacity=0.5,
            line=dict(width=3, color="DarkSlateGrey"),
        ),
        hovertemplate=None,
    )

    fig.update_xaxes(
        showticklabels=False,
        title="",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    fig.update_yaxes(
        showticklabels=False,
        title="",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    fig.update_layout(
        font=dict(
            family=" Arial, sans-serif",
            size=24,
            color="#363636",
        ),
        title_x=0.5,
        title_font_size=24,
    )

    return fig


languages_mapping = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Polish": "pl",
}

languages = list(languages_mapping.keys())

# Cached Ressources
client = get_client()
example_vectors_df = load_example_vectors()


# App Layout
st.title(":robot_face: Embeddings across Languages :earth_africa:")

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

st.expander(
    "How it Works",
    expanded=False,
).markdown(
    """
Type in some words in English and see how their embeddings compare across different languages.

GoogleTranslator is used for the translations, OpenAI's 'text-embedding-3-small' modsel to generate the embeddings and the first two principal components are used for visualization.

Best viewed in wide mode.

[GitHub Code](https://github.com/jkrol21/embeddings_exploration)
"""
)

user_input = st.text_input("English Words to Translate", INITIAL_WORDS_EXAMPLE)

trigger_button = st.button("Show Translation Embeddings")
# make sure that upon app loading example shown
trigger_button = max(trigger_button, user_input == INITIAL_WORDS_EXAMPLE)

if trigger_button and user_input.strip() != "":

    words = user_input.strip().replace(" ", ",").split(",")
    words = [word for word in words if len(word.strip()) > 0]

    if len(words) > 1:
        if len(words) > 20:
            st.info("Max 20 words at a time. Ommitting the rest.")
            words = words[:20]
        cmap = plt.get_cmap("tab10")
        word_colors = [cmap(i) for i in range(len(words))]

        first_row_plots = st.columns(COLS_PER_ROW)
        second_row_plots = st.columns(COLS_PER_ROW)

        for idx, language in enumerate(languages):
            language_code = languages_mapping[language]

            translated_words = (
                words
                if language == "English"
                else [translate(word, language_code) for word in words]
            )
            word_vectors = [get_word_embedding(word) for word in translated_words]

            language_fig = plot_pca_embeddings(
                translated_words,
                word_vectors,
                word_colors,
                title=language,
            )

            col_to_plot = (
                second_row_plots[idx % COLS_PER_ROW]
                if idx > 2
                else first_row_plots[idx]
            )
            with col_to_plot:
                st.plotly_chart(language_fig, use_container_width=True)

    else:
        st.error("At least 2 words needed.")
