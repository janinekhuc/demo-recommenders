""""Popularity based recommender."""


class PopularityRecommender():
    def __init__(self):
        self.data = None
        self.user_id = None
        self.item_id = None
        self.n_rec = None
        self.popularity_recommendations = None

    def create_recommendations(self, data, user_id, item_id, n_rec=15):
        """Create top n recommendations based on popularity of a given item."""
        self.data = data
        self.user_id = user_id
        self.item_id = item_id
        self.n_rec = n_rec

        data_grouped = data.groupby([self.item_id]).agg(
            {self.user_id: 'count'}).reset_index()
        data_grouped.rename(
            columns={self.user_id: 'score'}, inplace=True)

        data_sort = data_grouped.sort_values(
            ['score', self.item_id], ascending=[0, 1])
        data_sort['Rank'] = data_sort['score'].rank(
            ascending=0, method='first')
        self.popularity_recommendations = data_sort.head(n_rec)

    def recommend(self, user_id):
        """Recommend for an individual user."""
        user_recommendataion = self.popularity_recommendations
        user_recommendataion[self.user_id] = user_id
        cols = user_recommendataion.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendataion = user_recommendataion[cols]

        return user_recommendataion
