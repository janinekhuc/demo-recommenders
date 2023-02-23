from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Any


def check_matrix_sparsity(matrix):
    """Check matrix sparsity."""
    matrix_size = matrix.shape[0]*matrix.shape[1]
    interactions = len(matrix.nonzero()[0])
    sparsity = 100*(1 - (interactions/matrix_size))
    print("Matrix size: %s", matrix_size)
    print("Number of interactions: %s", interactions)
    print("Sparsity: %s", sparsity)


def get_label_map(df: pd.DataFrame, target_col: str):
    """Get the mapping between original label and its encoded value."""
    encoder = LabelEncoder()
    y_le = encoder.fit_transform(df[[target_col]])  # encode target variable
    # get the mapping between the original labels and encoded labels
    label_map = dict(zip(df[target_col], y_le))
    return label_map


def get_similarities(matrix):
    """Get similarities based cosine similaritiy matrix."""
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix


def get_tag_based_item_similarities(tags,
                                    tag_id_col_to_sort_by=cn.AU_item_ID_COL,
                                    tag_col=cn.AU_CATEGORIES_COL):
    """Get tag based cosine similaritiy matrix."""
    tags = tags.sort_values(by=tag_id_col_to_sort_by)
    tfidf_matrix = create_tfidf_matrix(tags, tag_col)
    similarity_matrix = get_similarities(tfidf_matrix)
    return similarity_matrix


def prepare_tfidf_vector(*args):
    """Prepare tfidf vector in english."""
    tfidf_vector = TfidfVectorizer(stop_words='english', *args)
    return tfidf_vector


def create_tfidf_matrix(df, cat_col, tfidf_vector=None):
    """Create tfidf matrix."""
    tfidf_vector = prepare_tfidf_vector() if tfidf_vector is None else tfidf_vector
    # apply the object to the cat column
    df[cat_col] = df[cat_col].str.replace(',', ' ')
    tfidf_matrix = tfidf_vector.fit_transform(df[cat_col])
    return tfidf_matrix
