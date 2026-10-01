import argparse
import csv
import logging
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import numpy as np


class CF_Model:
    def __init__(self, df_articles, df_behaviors, df_history):

        self.df_articles = df_articles
        self.df_behaviors = df_behaviors
        self.df_history = df_history
        self.df = pd.merge(df_behaviors, df_history, how="left", on="user_id")
        self.df_articles.set_index("article_id", inplace=True)

    def build_model(self):
        # Compute interaction matrix
        self.expanded_history = self.df_history.explode(
            [
                "article_id_fixed",
                "scroll_percentage_fixed",
                "read_time_fixed",
                "impression_time_fixed",
            ]
        )
        self.expanded_history["interaction"] = 1
        self.interaction_df = (
            self.expanded_history.groupby(["user_id", "article_id_fixed"])
            .agg(list)
            .reset_index()
        )
        self.interaction_df["interaction"] = self.interaction_df["interaction"].apply(
            lambda x: sum(x)
        )
        self.interaction_df["read_time_fixed"] = self.interaction_df[
            "read_time_fixed"
        ].apply(lambda x: sum(x))
        self.interaction_df["scroll_percentage_fixed"] = self.interaction_df[
            "scroll_percentage_fixed"
        ].apply(lambda x: max(x))
        self.interaction_df["score"] = (
            self.interaction_df["interaction"]
            + self.interaction_df["read_time_fixed"] / 120.0
            + self.interaction_df["scroll_percentage_fixed"] / 100.0
        )

        # Compute user similarity
        self.user_item_matrix = self.interaction_df.pivot_table(
            index="user_id", columns="article_id_fixed", values="score", fill_value=0
        )
        self.user_means = self.user_item_matrix.mean(axis=1)
        self.user_item_matrix_normalized = self.user_item_matrix.sub(
            self.user_means, axis=0
        )

        self.user_similarity = cosine_similarity(self.user_item_matrix_normalized)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

        # Use articles metadata to compute article similarity
        self.df_articles["topics"] = self.df_articles["topics"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df_articles["subcategory"] = self.df_articles["subcategory"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df_articles["category"] = self.df_articles["category"].fillna("unknown")
        self.df_articles[["total_pageviews", "total_read_time", "sentiment_score"]] = (
            self.df_articles[
                ["total_pageviews", "total_read_time", "sentiment_score"]
            ].fillna(0)
        )

        self.df_articles["topics"] = self.df_articles["topics"].apply(
            lambda x: " ".join(x)
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), ["category"]),
                (
                    "num",
                    StandardScaler(),
                    ["total_pageviews", "total_read_time", "sentiment_score"],
                ),
            ],
            remainder="drop",
        )

        self.metadata_normalized = self.preprocessor.fit_transform(
            self.df_articles[
                [
                    "topics",
                    "subcategory",
                    "category",
                    "total_pageviews",
                    "total_read_time",
                    "sentiment_score",
                ]
            ]
        )

        self.item_similarity = cosine_similarity(self.metadata_normalized)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.df_articles.index,
            columns=self.df_articles.index,
        )

    def predict_interaction(self, user_id, article_id):
        try:
            user_cf_prediction = 0
            if article_id in self.user_item_matrix.columns:
                users_who_interacted = self.user_item_matrix[
                    self.user_item_matrix[article_id] > 0
                ].index

                # User-based CF
                numerator_user_cf = sum(
                    self.user_similarity_df.loc[user_id, other_user]
                    * self.user_item_matrix.loc[other_user, article_id]
                    for other_user in users_who_interacted
                )
                denominator_user_cf = sum(
                    self.user_similarity_df.loc[user_id, other_user]
                    for other_user in users_who_interacted
                )

                if denominator_user_cf != 0:
                    user_cf_prediction = numerator_user_cf / denominator_user_cf

            # Item-based CF
            item_cf_prediction = 0

            interacted_items = self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] > 0
            ]
            numerator_item_cf = sum(
                self.item_similarity_df.loc[article_id, other_item]
                * self.user_item_matrix.loc[user_id, other_item]
                for other_item in interacted_items
            )
            denominator_item_cf = sum(
                self.item_similarity_df.loc[article_id, other_item]
                for other_item in interacted_items
            )

            if denominator_item_cf != 0:
                item_cf_prediction = numerator_item_cf / denominator_item_cf

            predicted_interaction = 0
            # Combine both predictions
            if user_cf_prediction != 0 and item_cf_prediction != 0:
                alpha = 0.5
                predicted_interaction = (alpha * user_cf_prediction) + (
                    (1 - alpha) * item_cf_prediction
                )
            elif user_cf_prediction != 0:
                predicted_interaction = user_cf_prediction
            elif item_cf_prediction != 0:
                predicted_interaction = item_cf_prediction
            # Adjust with user mean
            predicted_interaction += self.user_means.loc[user_id]

            return predicted_interaction
        except Exception:
            return 0

    def get_labels_and_scores(self, user_id, articles_inview, true_clicked_articles):
        predictions = [
            (article, self.predict_interaction(user_id, article))
            for article in articles_inview
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        true_labels = [
            1 if article in true_clicked_articles else 0 for article in articles_inview
        ]
        predicted_scores = [score for _, score in predictions]
        return true_labels, predicted_scores

    def compute_auc(self, user_id, articles_inview, true_clicked_articles):
        true_labels, predicted_scores = self.get_labels_and_scores(
            user_id, articles_inview, true_clicked_articles
        )
        if (
            len(set(true_labels)) > 1
        ):  # To ensure there's at least one positive and one negative sample
            return roc_auc_score(true_labels, predicted_scores)
        else:
            return 0.5  # AUC for a non-discriminative classifier

    def compute_mrr(self, user_id, articles_inview, true_clicked_articles):
        predictions = [
            (article, self.predict_interaction(user_id, article))
            for article in articles_inview
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        for rank, (article, score) in enumerate(predictions, start=1):
            if article in true_clicked_articles:
                return 1 / rank
        return 0

    def compute_dcg(self, relevance_scores, k):
        relevance_scores = np.asarray(relevance_scores)[:k]
        if relevance_scores.size:
            return np.sum(
                (2**relevance_scores - 1)
                / np.log2(np.arange(2, relevance_scores.size + 2))
            )
        return 0.0

    def compute_ndcg(self, user_id, articles_inview, true_clicked_articles, k):
        predictions = [
            (article, self.predict_interaction(user_id, article))
            for article in articles_inview
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        true_relevance = [
            1 if article in true_clicked_articles else 0 for article in articles_inview
        ]
        sorted_relevance = [
            true_relevance[list(articles_inview).index(article)]
            for article, _ in predictions
        ]

        dcg = self.compute_dcg(sorted_relevance, k)
        idcg = self.compute_dcg(sorted(true_relevance, reverse=True), k)

        return dcg / idcg if idcg > 0 else 0.0

    def compute_ndcg_at_5(self, user_id, articles_inview, true_clicked_articles):
        return self.compute_ndcg(user_id, articles_inview, true_clicked_articles, 5)

    def compute_ndcg_at_10(self, user_id, articles_inview, true_clicked_articles):
        return self.compute_ndcg(user_id, articles_inview, true_clicked_articles, 10)

    def evaluate_model(self, user_id, articles_inview, true_clicked_articles):
        auc = self.compute_auc(user_id, articles_inview, true_clicked_articles)
        mrr = self.compute_mrr(user_id, articles_inview, true_clicked_articles)
        ndcg_at_5 = self.compute_ndcg_at_5(
            user_id, articles_inview, true_clicked_articles
        )
        ndcg_at_10 = self.compute_ndcg_at_10(
            user_id, articles_inview, true_clicked_articles
        )

        return {"AUC": auc, "MRR": mrr, "NDCG@5": ndcg_at_5, "NDCG@10": ndcg_at_10}

    def evaluate_model_on_dataframe(self, df_evaluation):
        metrics_list = []

        for _, row in df_evaluation.iterrows():
            user_id = row["user_id"]
            articles_inview = row["article_ids_inview"]
            true_clicked_articles = row["article_ids_clicked"]

            metrics = self.evaluate_model(
                user_id, articles_inview, true_clicked_articles
            )
            metrics_list.append(metrics)

        # Calculate average metrics
        avg_metrics = {
            "AUC": np.mean([metrics["AUC"] for metrics in metrics_list]),
            "MRR": np.mean([metrics["MRR"] for metrics in metrics_list]),
            "NDCG@5": np.mean([metrics["NDCG@5"] for metrics in metrics_list]),
            "NDCG@10": np.mean([metrics["NDCG@10"] for metrics in metrics_list]),
        }

        return avg_metrics

    def rank_articles(self, df_behaviors):
        results = []
        for _, row in df_behaviors.iterrows():
            user_id = row["user_id"]
            articles_inview = row["article_ids_inview"]

            predictions = [
                (article, self.predict_interaction(user_id, article))
                for article in articles_inview
            ]
            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

            ranks_dict = {
                article: rank + 1
                for rank, (article, _) in enumerate(sorted_predictions)
            }

            ranks = [ranks_dict[article] for article in articles_inview]

            results.append({"user_id": user_id, "article_ids_inview": ranks})

        df_ranking = pd.DataFrame(results)
        df_ranking["article_ids_inview"] = df_ranking["article_ids_inview"].apply(
            lambda x: str(x).replace(" ", "")
        )
        df_ranking.to_csv(
            "./results/collaborative_filtering_predictions.txt",
            header=False,
            index=False,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        return df_ranking


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Collaborative filtering model script. This script builds the model using the training split of the data and outputs predictions based on the validation split of the data."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to the data folder contaning the parquet files",
    )
    parser.add_argument(
        "--use_validation_split",
        required=False,
        type=bool,
        default=True,
        help="Flag that specifies whether to use the validation split of the data or another behaviors file. If set to False the user must provide the --test_set parameter",
    )
    parser.add_argument(
        "--test_set",
        required=False,
        type=str,
        help="Path to the test set to be predicted",
    )
    args = parser.parse_args()
    PATH = Path(args.data_folder)

    logging.info("Step 1/3: Reading dataset")

    df_behaviors_train = pd.read_parquet(PATH.joinpath("train", "behaviors.parquet"))
    df_history_train = pd.read_parquet(PATH.joinpath("train", "history.parquet"))
    df_articles = pd.read_parquet(PATH.joinpath("articles.parquet"))
    df_behaviors_val = pd.read_parquet(PATH.joinpath("validation", "behaviors.parquet"))
    df_history_val = pd.read_parquet(PATH.joinpath("validation", "history.parquet"))

    logging.info("Step 2/3: Building model")
    model = CF_Model(df_articles, df_behaviors_train, df_history_train)
    model.build_model()

    logging.info("Step 3/3: Generating predictions")

    if args.use_validation_split:
        result = model.rank_articles(df_behaviors_val)
    else:
        df_test = pd.read_parquet(args.test_set)
        result = model.rank_articles(df_test)
