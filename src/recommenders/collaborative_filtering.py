import pandas as pd
import numpy as np
import typing as t
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from recommenders.general_utils import (get_similarities,
                                        get_tag_based_item_similarities,
                                        get_label_map)
from recommenders.utils import exclusive_or

def create_user_item_matrix(user_ids, item_ids):
    """Create user-item matrix that serves as placeholder for similarities."""
    user_item_matrix = csr_matrix(([1]*len(user_ids), (item_ids, user_ids)))
    return user_item_matrix


def get_user_item_similarities(user_item_matrix):
    """Get user- item based cosine similaritiy matrix."""
    similarity_matrix = get_similarities(user_item_matrix)
    return similarity_matrix


def get_user_item_tag_based_similarities(user_item_matrix, all_items,
                                         tag_id_col_to_sort_by,
                                         tag_col):
    """Get user- item and tag_based item cosine similarities based on equal weights."""
    user_item_similarities = get_user_item_similarities(user_item_matrix)
    tag_based_item_similarities = get_tag_based_item_similarities(
        all_items, tag_id_col_to_sort_by, tag_col)
    similarity_matrix = (tag_based_item_similarities+user_item_similarities)/2
    return similarity_matrix


def get_similarity_method(method: t.Literal['tag_based', 'combined', 'user_item_based'],
                          user_item_matrix: pd.DataFrame = None, all_items: pd.DataFrame = None,
                          tag_col: str = None):
    """Depending on the required method, create a similarity matrix."""
    if method == 'tag_based':
        print('Creating tag_based similarities')
        similarity_matrix = get_tag_based_item_similarities(all_items)
    elif method == 'combined':
        similarity_matrix = get_user_item_tag_based_similarities(
            user_item_matrix, all_items, tag_col=tag_col)
    else:
        similarity_matrix = get_user_item_similarities(user_item_matrix)
    return similarity_matrix


class CollabFiltering():
    def __init__(self) -> None:
        self.data = None
        self.user_col = None
        self.item_col = None
        self.user_id = None
        self.user_index = None
        self.n_rec = None

        # recommendation related matrices
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.sparse_user_item_matrix = None
        self.user_item_scores = None

        # encoders
        self.user_label_encoder = None
        self.user_ids = None
        self.item_label_encoder = None
        self.item_ids = None

    def _set_encoders(self, data, user_col, item_col):
        """Set encoders to remap from index to strings."""
        self.user_label_encoder = LabelEncoder()
        self.user_ids = self.user_label_encoder.fit_transform(
            data[user_col])
        self.item_label_encoder = LabelEncoder()
        self.item_ids = self.item_label_encoder.fit_transform(
            data[item_col])

    def create_recommendations(self, data:pd.DataFrame, user_col:str,
                               item_col:str, all_items=None,
                               method: t.Literal['tag_based', 'combined',
                                                 'user_item_based'] = 'tag_based',
                               tag_col:str):
        """Create the recommendation matrices."""
        self.data = data.sort_values(
            by=item_col)  # sort to ensure corsrect order also for tags
        self.user_col = user_col
        self.item_col = item_col
        self._set_encoders(data, self.user_col, self.item_col)

        # compute recommendations/ similarities of items based on user likes
        self.user_item_matrix = create_user_item_matrix(
            self.user_ids, self.item_ids)
        self.similarity_matrix = get_similarity_method(
            method, self.user_item_matrix, all_items=all_items, tag_col=tag_col)

        self.sparse_user_item_matrix = csr_matrix(self.user_item_matrix.T)
        self.user_item_scores = self.sparse_user_item_matrix.dot(
            self.similarity_matrix)

    def _recommend_based_on_user_index(self):
        """Recommend based on a user index."""
        scores = self.user_item_scores[self.user_index, :]
        # # get items not to be recommended as already liked
        liked_item_indexes = self._get_liked_item_indexes(self.user_index)
        # retrain model with different weights if not enough likes?
        scores = self._remove_liked_items(
            scores, self.user_index, liked_item_indexes)
        top_item_ids = self._sort_top_items(scores, self.n_rec)
        recommendations = self._create_top_user_recommendations(
            top_item_ids, scores)
        recommendations[cn.LIKED_itemS_COL] = self.n_rec * [self._get_liked_items(
            liked_item_indexes).tolist()]
        return recommendations

    def recommend(self, n_rec: int, user_id: str = None,
                  user_index: int = None, for_eval: bool = False):
        """Create recommendations for individual user."""
        if not exclusive_or(user_id is not None, user_index is not None):
            raise ValueError(
                'Can only create recommendation by user id OR user index')
        self.n_rec = n_rec
        self.user_index = self._get_user_index(
            user_id) if user_index is None else user_index
        recommendations = self._recommend_based_on_user_index()
        if for_eval:
            recommendations = self._remap_ids_to_names(recommendations)
        return recommendations

    def _remap_ids_to_names(self, df: pd.DataFrame, item_id_col: str , item_title_col: str):
        """Remap item_id_col to readable item_title_col. Mostly used for evaluation purposes."""
        id_to_name_map = dict(
            zip(self.data[item_id_col], self.data[item_title_col]))
        df[item_id_col] = df[item_id_col].map(id_to_name_map)
        df[cn.LIKED_itemS_COL] = df[cn.LIKED_itemS_COL].map(
            lambda x: list(map(id_to_name_map.get, x)))
        return df

    def _get_user_index(self, user_id):
        """Given a user_id column, get the user_index."""
        user_index_dict = get_label_map(self.data, self.user_col)
        return user_index_dict[user_id]

    def _get_liked_item_indexes(self, user_index):
        """Get liked item indexes."""
        return self.sparse_user_item_matrix.indices[self.sparse_user_item_matrix.indptr[user_index]:
                                                    self.sparse_user_item_matrix.indptr[user_index+1]]

    def _get_liked_items(self, liked_item_indexes):
        """Get liked items."""
        return self.item_label_encoder.inverse_transform(liked_item_indexes)

    def _remove_liked_items(self,  scores, user_index, liked_item_indexes=None):
        """Remove liked items, assuming they are indicated as -1s."""
        if liked_item_indexes is None:
            liked_item_indexes = self._get_liked_item_indexes(user_index)
        # do not recommend already liked items
        scores[liked_item_indexes] = -1
        return scores

    def _sort_top_items(self, scores, n_rec):
        """Get top n items."""
        return np.argsort(scores)[-n_rec:][::-1]

    def _create_top_user_recommendations(self, top_item_ids, scores, user_index=None):
        """Create a pd.DataFrame containing the top recommendations, with score, rank, user_id and items names."""
        user_index = self.user_index if user_index is None else user_index
        recs = pd.DataFrame({
            self.item_col: top_item_ids,
            'scores': scores[top_item_ids],
            'rank': range(1, len(top_item_ids) + 1)
        }
        )
        recs[self.user_col] = user_index

        # remap userid & content
        recs[self.item_col] = self.item_label_encoder.inverse_transform(
            recs[self.item_col])
        recs[self.user_col] = self.user_label_encoder.inverse_transform(
            recs[self.user_col])
        return recs

    def recommend_for_all(self, n_rec, for_eval):
        """Create recommendations for entire userbase."""
        all_recommendations = pd.DataFrame()
        for user_index in range(len(self.user_item_scores)):
            recommendations = self.recommend(n_rec=n_rec,
                                             user_index=user_index,
                                             for_eval=for_eval)
            all_recommendations = pd.concat(
                [all_recommendations, recommendations])
        return all_recommendations
