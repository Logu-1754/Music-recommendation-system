# music_recommender.py
"""
Music Recommendation System (toy/small demo)
- Generates a small synthetic dataset (songs, users, ratings)
- Implements:
    1) User-based Collaborative Filtering (cosine similarity)
    2) Content-based recommendations (TF-IDF on metadata)
    3) Hybrid combination of both
- Evaluates CF using RMSE on a test split
- Demonstrates top-N recommendations for a sample user
Run:
    python music_recommender.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def generate_synthetic_data(n_users=50, n_songs=200):
    artists = [f'Artist {i}' for i in range(1, 31)]
    genres_list = ['pop','rock','hiphop','jazz','classical','electronic','folk','country','blues']
    tags_list = ['chill','dance','party','love','sad','energetic','instrumental','acoustic','upbeat','melancholy']

    songs = []
    for i in range(1, n_songs+1):
        title = f"Song {i}"
        artist = np.random.choice(artists)
        genre_count = np.random.randint(1, 3)
        genres = ','.join(np.random.choice(genres_list, size=genre_count, replace=False))
        tag_count = np.random.randint(1, 3)
        tags = ','.join(np.random.choice(tags_list, size=tag_count, replace=False))
        songs.append({'song_id': i, 'title': title, 'artist': artist, 'genres': genres, 'tags': tags})

    songs_df = pd.DataFrame(songs)

    users = [{'user_id': u, 'name': f'User {u}'} for u in range(1, n_users+1)]
    users_df = pd.DataFrame(users)

    # assign each user 2 favorite genres (to bias ratings)
    user_favs = {u['user_id']: list(np.random.choice(genres_list, size=2, replace=False)) for u in users}

    ratings = []
    for u in users_df['user_id']:
        # number of songs rated by this user
        n_rate = max(5, np.random.poisson(18))  # ensures at least a few ratings per user
        rated_songs = np.random.choice(songs_df['song_id'], size=min(n_rate, len(songs_df)), replace=False)
        for s in rated_songs:
            song = songs_df.loc[songs_df['song_id'] == s].iloc[0]
            base = 3.0
            song_genres = song['genres'].split(',')
            # boost rating if song includes a user's favorite genre
            if any(g in user_favs[u] for g in song_genres):
                mean = base + 1.0
            else:
                mean = base - 0.4
            rating = float(np.clip(np.round(np.random.normal(loc=mean, scale=0.8)), 1, 5))
            ratings.append({'user_id': int(u), 'song_id': int(s), 'rating': rating})

    ratings_df = pd.DataFrame(ratings)
    return users_df, songs_df, ratings_df

# ---------------------------
# Content-based (TF-IDF)
# ---------------------------
def build_tfidf_similarity(songs_df):
    # combine metadata fields into one text
    songs_df = songs_df.copy()
    songs_df['metadata'] = songs_df[['title', 'artist', 'genres', 'tags']].astype(str).agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    song_matrix = tfidf.fit_transform(songs_df['metadata'])
    song_sim = cosine_similarity(song_matrix, song_matrix)  # song x song
    return song_sim, tfidf

# ---------------------------
# Collaborative Filtering (user-based)
# ---------------------------
def train_user_item_matrix(train_df, users, songs):
    user_item = train_df.pivot(index='user_id', columns='song_id', values='rating')
    # keep all users and songs (some columns may be missing if not rated in train)
    user_item = user_item.reindex(index=users['user_id'], columns=songs['song_id'])
    return user_item

def build_user_similarity(user_item):
    # for similarity, fill NaN with 0 (treat missing as 0 for vector similarity)
    filled = user_item.fillna(0).values
    user_sim = cosine_similarity(filled)  # user x user
    return user_sim

def predict_rating_user_based(user_id, song_id, user_item, user_sim):
    # user_item: DataFrame indexed by user_id, columns song_id (ratings with NaN for unknowns)
    user_ids = list(user_item.index)
    song_ids = list(user_item.columns)

    # fallback global mean
    global_mean = user_item.stack().mean()

    if user_id not in user_ids and song_id not in song_ids:
        return global_mean

    # if user unknown
    if user_id not in user_ids:
        # return item mean if possible else global
        if song_id in song_ids:
            col = user_item[song_id]
            if col.count() > 0:
                return col.mean()
        return global_mean

    u_idx = user_ids.index(user_id)

    if song_id not in song_ids:
        # new song -> return user's mean if available else global
        row = user_item.loc[user_id]
        if row.count() > 0:
            return row.mean()
        return global_mean

    # users who rated this song
    col = user_item[song_id]
    rated_mask = ~col.isna()
    if rated_mask.sum() == 0:
        # item has no ratings in train -> return user mean or global
        user_row = user_item.loc[user_id]
        return user_row.mean() if user_row.count() > 0 else global_mean

    sims = user_sim[u_idx]  # similarities of target user to all users
    # only keep users who rated the song
    sims = sims[rated_mask.values]
    ratings = col[rated_mask].values
    other_user_ids = user_item.index[rated_mask]

    # subtract mean-centering
    other_means = user_item.loc[other_user_ids].mean(axis=1).values
    numerator = np.dot(sims, ratings - other_means)
    denom = np.sum(np.abs(sims))
    user_mean = user_item.loc[user_id].mean() if user_item.loc[user_id].count() > 0 else user_item.stack().mean()
    if denom == 0:
        return user_mean
    pred = user_mean + numerator / denom
    pred = float(np.clip(pred, 1.0, 5.0))
    return pred

# ---------------------------
# Hybrid recommendation: combine CF + content
# ---------------------------
def content_score_for_user(user_id, candidate_song_id, user_item, song_sim, song_index_map):
    # look at songs the user liked (rating >= 4) in training data
    if user_id not in user_item.index or candidate_song_id not in user_item.columns:
        return 0.0
    user_row = user_item.loc[user_id]
    liked_mask = user_row >= 4.0
    liked_songs = user_row[liked_mask].index.tolist()
    if len(liked_songs) == 0:
        return 0.0
    j = song_index_map[candidate_song_id]
    liked_indices = [song_index_map[s] for s in liked_songs if s in song_index_map]
    if len(liked_indices) == 0:
        return 0.0
    sims = song_sim[j, liked_indices]
    # weight by (rating - 3) to give stronger weight to 5s
    weights = (user_row[liked_songs].values - 3.0)
    if weights.sum() == 0:
        return float(np.mean(sims))
    score = np.dot(sims, weights) / np.sum(np.abs(weights))
    return float(score)

def recommend_for_user(user_id, user_item, user_sim, song_sim, songs_df, top_n=10, alpha=0.7):
    """
    alpha: weight for CF (0..1). Hybrid score = alpha*CF_score_norm + (1-alpha)*content_score_norm
    """
    all_song_ids = list(user_item.columns)
    rated_by_user = user_item.loc[user_id].dropna().index.tolist() if user_id in user_item.index else []
    candidates = [s for s in all_song_ids if s not in rated_by_user]

    song_index_map = {sid: idx for idx, sid in enumerate(songs_df['song_id'].tolist())}

    # compute raw CF scores
    cf_scores = {}
    for s in candidates:
        cf_scores[s] = predict_rating_user_based(user_id, s, user_item, user_sim)

    # normalize CF into 0-1
    cf_vals = np.array(list(cf_scores.values()))
    cf_min, cf_max = cf_vals.min(), cf_vals.max()
    if cf_max - cf_min > 0:
        for k in cf_scores:
            cf_scores[k] = (cf_scores[k] - cf_min) / (cf_max - cf_min)
    else:
        for k in cf_scores:
            cf_scores[k] = 0.5

    # compute content scores
    content_scores = {}
    for s in candidates:
        content_scores[s] = content_score_for_user(user_id, s, user_item, song_sim, song_index_map)
    cont_vals = np.array(list(content_scores.values()))
    cont_min, cont_max = cont_vals.min(), cont_vals.max()
    if cont_max - cont_min > 0:
        for k in content_scores:
            content_scores[k] = (content_scores[k] - cont_min) / (cont_max - cont_min)
    else:
        for k in content_scores:
            content_scores[k] = 0.0

    # hybrid combine
    hybrid = []
    for s in candidates:
        score = alpha * cf_scores[s] + (1 - alpha) * content_scores[s]
        hybrid.append((s, score))
    hybrid_sorted = sorted(hybrid, key=lambda x: x[1], reverse=True)[:top_n]
    # return song details
    recs = []
    for sid, sc in hybrid_sorted:
        row = songs_df[songs_df['song_id'] == sid].iloc[0]
        recs.append({'song_id': sid, 'title': row['title'], 'artist': row['artist'],
                     'genres': row['genres'], 'score': sc})
    return recs

# ---------------------------
# Evaluation (RMSE on test set using CF predictions)
# ---------------------------
def evaluate_rmse(test_df, user_item, user_sim):
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        user = int(row['user_id'])
        song = int(row['song_id'])
        true = float(row['rating'])
        pred = predict_rating_user_based(user, song, user_item, user_sim)
        y_true.append(true)
        y_pred.append(pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# ---------------------------
# Main flow
# ---------------------------
def main():
    print("Generating synthetic data...")
    users_df, songs_df, ratings_df = generate_synthetic_data(n_users=50, n_songs=200)
    print(f"Users: {len(users_df)}, Songs: {len(songs_df)}, Ratings: {len(ratings_df)}")

    # split into train/test
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")

    # prepare user-item matrix
    user_item = train_user_item_matrix(train_df, users_df, songs_df)
    print("Built user-item matrix:", user_item.shape)

    # user similarity
    user_sim = build_user_similarity(user_item)
    print("Computed user-user similarity.")

    # content similarity
    song_sim, tfidf = build_tfidf_similarity(songs_df)
    print("Computed song-song similarity (TF-IDF).")

    # Evaluate CF
    rmse = evaluate_rmse(test_df, user_item, user_sim)
    print(f"CF RMSE on test set: {rmse:.4f}")

    # Demo: recommend for a sample user
    sample_user = 1
    if sample_user not in user_item.index:
        sample_user = user_item.index[0]
    print(f"\nTop recommendations for user {sample_user}:")
    recs = recommend_for_user(sample_user, user_item, user_sim, song_sim, songs_df, top_n=10, alpha=0.75)
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['title']} — {r['artist']} (genres: {r['genres']}) score: {r['score']:.3f}")

    # optionally save csv files for pushing to repo
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    users_df.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    songs_df.to_csv(os.path.join(out_dir, "songs.csv"), index=False)
    ratings_df.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)
    print(f"\nSaved users/songs/ratings CSVs under {out_dir}/ for inspection or upload to GitHub.")

if __name__ == "__main__":
    main()
