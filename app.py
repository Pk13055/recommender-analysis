"""
    :author: pk13055
    :brief: Building a recommender system
"""
import os
import glob

import numpy as np
import pandas as pd
import streamlit as st


@st.cache()
def get_data(filename: str, index_col: int = 0, parse_dates: bool = True):
    """Load data csv into dataframe"""
    df = pd.read_csv(filename, index_col=0)
    return df


def get_onehot(df: pd.DataFrame, return_genres=False) -> pd.DataFrame:
    """Create a one-hot encoding of genres"""
    try:
        genres = list(set.union(*df.genres.str.split("|").apply(set).tolist()))
        one_hot = np.vstack(df.genres.map(lambda genre_str: np.array([int(genre in genre_str) for genre in genres])).to_numpy())
        genre_df = pd.DataFrame(one_hot, columns=genres, index=df.index)
    except Exception as e:
        st.error(f"Could not coarse `genres` | {e}")
        genres = []
        genre_df = pd.DataFrame()
    finally:
        if return_genres:
            return genre_df, genres
        return genre_df


@st.cache()
def get_dataset(df: pd.DataFrame, genre_df: pd.DataFrame) -> pd.DataFrame:
    """Returns aggegrated dataset with all basic fields"""
    dataset = pd.concat([df, pd.read_csv("data/links.csv", index_col=0), genre_df], axis=1)
    del dataset['genres']
    del dataset['title']
    return dataset


@st.cache()
def get_ratings(dataset: pd.DataFrame, genres: list) -> pd.DataFrame:
    """Add average rating and count values"""
    rating = pd.read_csv("data/ratings.csv")
    rating_mean = rating.groupby('movieId').mean().rating.rename('rating')
    rating_count = rating.groupby('movieId').count().rating.rename('co')
    _df = pd.concat([rating_mean, rating_count, dataset], axis=1)
    cols = set(list(_df.columns))
    return _df.loc[:, list(cols - set(genres)) + genres].dropna()


def main():
    st.title("Recommender System")
    st.header("Loading the data")

    """
        We're going to be using the
        [movie lens dataset](https://grouplens.org/datasets/movielens/25m/)
        for the analysis.
    """

    st.sidebar.info("Choose the original `movies.csv`")
    filename = st.sidebar.selectbox(
        "Choose first dataset",
        glob.glob("data/*.csv"),
    )

    df = get_data(filename)

    st.sidebar.info("Choose the original `rating.csv`")
    filename_rating = st.sidebar.selectbox(
        "Choose second dataset",
        glob.glob("data/*.csv"),
    )

    "Here's a sample sample of the raw dataset"
    st.table(df.head())
    f"Total of ${df.shape[0]}$ records"

    st.subheader("Year Extraction")
    """
        The first step is to include a year by
        extracting it from the title (_and subsequently remove the year from the title_)
    """

    df.loc[:, 'name'] = df.title.str.replace(r'\(\d{4}\)', '').str.strip()
    df.loc[:, 'year'] = df.title.str.extract(r'(\d{4})', expand=False)
    st.dataframe(df.head(10))

    st.subheader("One Hot Encoding")
    """
                    The next step is to create a one-hot encoding for the genres
                    Consider a function $F(x)$, where $x \in X$, where $X$ is
                    the set of all possible genre combinations. Now, $F$ is defined as:
                    $$
                        F: X \\rightarrow Y,F(x \in X) = y \in Y
                    $$
                    where $y$ is defined as a vector such that $y(i) = 1$ if a
                    genre, x(i) is present for a movie. Thus the elements of $y$
                    act as a bitmask for the set of genres $G$, such that all $x(i) \in G$
    """
    genre_df, genres = get_onehot(df, return_genres=True)

    "$x(i) \in G, G =$"
    genres

    "The one hot encoding, thus, is something like this"
    st.table(genre_df.head(10).T)
    f"`Actual shape: {genre_df.shape}`"

    """
        Modifying the original dataframe by including this new genre matrix,
        (additionally, just joining the `links.csv` data), we have our initial dataset
    """
    dataset = get_dataset(df, genre_df)
    st.dataframe(dataset.head(10))
    cols = list(dataset.columns)
    st.write(f"This dataset has ${dataset.shape[0]}$ records and {cols} fields")

    st.header("Additional Features")
    st.subheader("Average rating and user count")

    final_df = get_ratings(dataset, genres)
    st.dataframe(final_df.head(10))
    f"The final prepared dataset has ${final_df.shape[0]}$ records"

    st.subheader("Fetching additional information using `imdbId`")

    st.write("`TODO: add imdb fetching of details`")

    st.header("Visualization")

    # TODO: visualization is to be added here
    # first choose two subsets of data, ie, a different combination of genres
    # (NOTE use `joinplot` for this)
    # https://seaborn.pydata.org/generated/seaborn.jointplot.html
    # - year vs rating
    # - year vs count
    # - rating vs count
    # - Additional info (from imdb) vs existing fields
    # (NOTE: the genre based comparisons can also be done)


if __name__ == "__main__":
    main()

