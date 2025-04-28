import pandas as pd
import numpy as np
import Recommenders as Recommenders
import EnhancedRecommenders as EnhancedRecommenders
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from functools import lru_cache
from scipy.sparse import csr_matrix

print("Starting the recommendation system...")
start_time = time.time()

# Load the data
print("\nLoading data files...")
load_start = time.time()
triplets_file = 'triplets_file/triplets_file.csv'
song_metadata_file = 'song_data/song_data.csv'

# Read the data with low_memory=False to handle mixed types
print("Reading triplets file...")
song_df_1 = pd.read_csv(triplets_file, header=None, low_memory=False)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

# Convert listen_count to numeric
song_df_1['listen_count'] = pd.to_numeric(song_df_1['listen_count'], errors='coerce')

print("Reading song metadata file...")
song_df_2 = pd.read_csv(song_metadata_file, low_memory=False)

# Take a subset of the data before merging to reduce memory usage
print("\nTaking a subset of the data for evaluation...")
n_users = 1000  # Number of users to include
user_subset = song_df_1['user_id'].value_counts().head(n_users).index
song_df_1_subset = song_df_1[song_df_1['user_id'].isin(user_subset)]

print("Merging data...")
song_df = pd.merge(song_df_1_subset, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

# Enhanced Data Preprocessing
print("\nPreprocessing data...")
# Remove duplicates
song_df = song_df.drop_duplicates(['user_id', 'song_id'])

# Normalize listen counts
scaler = MinMaxScaler()
song_df['normalized_listen_count'] = scaler.fit_transform(song_df[['listen_count']].fillna(0))

# Create sparse user-song interaction matrix
print("Creating user-song interaction matrix...")
user_ids = song_df['user_id'].astype('category').cat.codes
song_ids = song_df['song_id'].astype('category').cat.codes
values = song_df['normalized_listen_count'].values

user_song_matrix = csr_matrix((values, (user_ids, song_ids)))

load_time = time.time() - load_start
print(f"Data loading and preprocessing completed in {load_time:.2f} seconds")

# Split data into train and test sets
print("\nSplitting data into train and test sets...")
train_data, test_data = train_test_split(song_df, test_size=0.2, random_state=42)

# Create recommenders
print("\nCreating recommendation models...")
model_start = time.time()

# Create base recommenders with caching
class CachedPopularityRecommender(Recommenders.popularity_recommender_py):
    @lru_cache(maxsize=1000)
    def recommend(self, user_id):
        return super().recommend(user_id)

class CachedItemSimilarityRecommender(Recommenders.item_similarity_recommender_py):
    @lru_cache(maxsize=1000)
    def recommend(self, user):
        return super().recommend(user)
    
    @lru_cache(maxsize=1000)
    def get_similar_items(self, item_list):
        return super().get_similar_items(item_list)

class CachedContentBasedRecommender(Recommenders.content_based_recommender_py):
    @lru_cache(maxsize=1000)
    def recommend(self, user_id, n_recommendations=10):
        return super().recommend(user_id, n_recommendations)
    
    @lru_cache(maxsize=1000)
    def get_song_similarity(self, song_id, n_recommendations=10):
        return super().get_song_similarity(song_id, n_recommendations)

# Initialize base recommenders
print("Initializing popularity-based recommender...")
pr = CachedPopularityRecommender()
pr.create(train_data, 'user_id', 'song_id')

print("Initializing item similarity recommender...")
ir = CachedItemSimilarityRecommender()
ir.create(train_data, 'user_id', 'song_id')

print("Initializing content-based recommender...")
cr = CachedContentBasedRecommender()
cr.create(train_data, 'user_id', 'song_id', song_df_2)

print("Initializing hybrid recommender...")
# Initialize enhanced recommender
hr = EnhancedRecommenders.HybridRecommender(
    popularity_weight=0.3,
    item_similarity_weight=0.3,
    content_weight=0.2,
    artist_weight=0.2
)
hr.create(train_data, song_df_2)

# Save models for later use
print("Saving models...")
joblib.dump(pr, 'popularity_recommender.joblib')
joblib.dump(ir, 'item_similarity_recommender.joblib')
joblib.dump(cr, 'content_based_recommender.joblib')
joblib.dump(hr, 'hybrid_recommender.joblib')

model_time = time.time() - model_start
print(f"Model creation completed in {model_time:.2f} seconds")

def calculate_recommendation_metrics(recommender, test_data, k=10, year_range=None, genre=None):
    """
    Calculate recommendation metrics including F1 score, precision, recall, and hit rate
    """
    total_precision = 0
    total_recall = 0
    total_hits = 0
    total_users = 0
    
    # Get unique users from test set
    test_users = test_data['user_id'].unique()
    
    for user in test_users[:100]:  # Reduced to 100 users for faster evaluation
        # Get actual items in test set for this user
        actual_items = set(test_data[test_data['user_id'] == user]['song_id'])
        
        if len(actual_items) == 0:
            continue
            
        try:
            # Get recommendations
            if isinstance(recommender, CachedPopularityRecommender):
                recommendations = recommender.recommend(user)
                if recommendations is None or recommendations.empty:
                    continue
                recommended_items = set(recommendations['song_id'].head(k))
            elif isinstance(recommender, CachedItemSimilarityRecommender):
                # For item similarity recommender
                user_items = recommender.get_user_items(user)
                if len(user_items) == 0:
                    continue
                recommendations = recommender.recommend(user)
                if recommendations is None or recommendations.empty:
                    continue
                # Handle the 'song' column name
                recommendations = recommendations.rename(columns={'song': 'song_id'})
                recommended_items = set(recommendations['song_id'].head(k))
            elif isinstance(recommender, CachedContentBasedRecommender):
                recommendations = recommender.recommend(user, k)
                if recommendations is None or recommendations.empty:
                    continue
                recommended_items = set(recommendations['song_id'].head(k))
            else:  # Hybrid recommender
                recommendations = recommender.recommend(user, k, year_range, genre)
                if recommendations is None or recommendations.empty:
                    continue
                recommended_items = set(recommendations['song_id'].head(k))
            
            # Calculate metrics
            if len(recommended_items) > 0:
                true_positives = len(actual_items.intersection(recommended_items))
                
                # Calculate precision and recall
                precision = true_positives / len(recommended_items)
                recall = true_positives / len(actual_items)
                
                # Calculate hit rate (if at least one recommendation is correct)
                hit_rate = 1 if true_positives > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_hits += hit_rate
                total_users += 1
                
        except Exception as e:
            print(f"Error processing user {user}: {str(e)}")
            continue
    
    if total_users == 0:
        return 0, 0, 0, 0
    
    avg_precision = total_precision / total_users
    avg_recall = total_recall / total_users
    avg_hit_rate = total_hits / total_users
    
    # Calculate F1 score
    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0
        
    return f1_score, avg_precision, avg_recall, avg_hit_rate

# Calculate metrics
print("\nCalculating recommendation metrics...")
metrics_start = time.time()

try:
    # Calculate for popularity-based recommender
    print("Evaluating popularity-based recommender...")
    f1_pop, prec_pop, rec_pop, hit_pop = calculate_recommendation_metrics(pr, test_data)
    print(f"\nPopularity-based Recommender:")
    print(f"F1 Score: {f1_pop:.4f}")
    print(f"Precision: {prec_pop:.4f}")
    print(f"Recall: {rec_pop:.4f}")
    print(f"Hit Rate: {hit_pop:.4f}")

    # Calculate for item-similarity recommender
    print("\nEvaluating item similarity recommender...")
    f1_item, prec_item, rec_item, hit_item = calculate_recommendation_metrics(ir, test_data)
    print(f"\nItem-Similarity Recommender:")
    print(f"F1 Score: {f1_item:.4f}")
    print(f"Precision: {prec_item:.4f}")
    print(f"Recall: {rec_item:.4f}")
    print(f"Hit Rate: {hit_item:.4f}")

    # Calculate for content-based recommender
    print("\nEvaluating content-based recommender...")
    f1_content, prec_content, rec_content, hit_content = calculate_recommendation_metrics(cr, test_data)
    print(f"\nContent-Based Recommender:")
    print(f"F1 Score: {f1_content:.4f}")
    print(f"Precision: {prec_content:.4f}")
    print(f"Recall: {rec_content:.4f}")
    print(f"Hit Rate: {hit_content:.4f}")

    # Calculate for hybrid recommender
    print("\nEvaluating hybrid recommender...")
    f1_hybrid, prec_hybrid, rec_hybrid, hit_hybrid = calculate_recommendation_metrics(hr, test_data)
    print(f"\nHybrid Recommender:")
    print(f"F1 Score: {f1_hybrid:.4f}")
    print(f"Precision: {prec_hybrid:.4f}")
    print(f"Recall: {rec_hybrid:.4f}")
    print(f"Hit Rate: {hit_hybrid:.4f}")

except Exception as e:
    print(f"\nError calculating metrics: {str(e)}")
    print("Continuing with the rest of the examples...")

metrics_time = time.time() - metrics_start
print(f"\nMetrics calculation completed in {metrics_time:.2f} seconds")

# Example 1: Get hybrid recommendations for a specific user
print("\nExample 1: Hybrid Recommendations for a Specific User")
print("---------------------------------------------------")
rec_start = time.time()
# Get a valid user_id from the dataset
user_id = song_df['user_id'].iloc[0]
hybrid_recommendations = hr.recommend(user_id)
# Merge with song metadata to get titles
hybrid_recommendations = pd.merge(hybrid_recommendations, song_df[['song_id', 'title', 'artist_name']].drop_duplicates(), on='song_id')
rec_time = time.time() - rec_start
print(f"\nHybrid recommendations for user {user_id}")
print(hybrid_recommendations[['song_id', 'title', 'artist_name', 'score']].head())
print(f"Recommendation time: {rec_time:.2f} seconds")

# Example 2: Get genre-specific recommendations
print("\nExample 2: Genre-Specific Recommendations")
print("---------------------------------------")
rec_start = time.time()
# Get a valid user_id and genre
user_id = song_df['user_id'].iloc[0]
genre = 0  # First genre cluster
genre_recommendations = hr.recommend(user_id, genre=genre)
# Merge with song metadata to get titles
genre_recommendations = pd.merge(genre_recommendations, song_df[['song_id', 'title', 'artist_name']].drop_duplicates(), on='song_id')
rec_time = time.time() - rec_start
print(f"\nGenre {genre} recommendations for user {user_id}")
print(genre_recommendations[['song_id', 'title', 'artist_name', 'score']].head())
print(f"Recommendation time: {rec_time:.2f} seconds")

# Example 3: Get year-filtered recommendations
print("\nExample 3: Year-Filtered Recommendations")
print("--------------------------------------")
rec_start = time.time()
# Get a valid user_id and year range
user_id = song_df['user_id'].iloc[0]
year_range = (2000, 2010)  # Songs from 2000 to 2010
year_recommendations = hr.recommend(user_id, year_range=year_range)
# Merge with song metadata to get titles
year_recommendations = pd.merge(year_recommendations, song_df[['song_id', 'title', 'artist_name', 'year']].drop_duplicates(), on='song_id')
rec_time = time.time() - rec_start
print(f"\nYear-filtered recommendations for user {user_id}")
print(year_recommendations[['song_id', 'title', 'artist_name', 'year', 'score']].head())
print(f"Recommendation time: {rec_time:.2f} seconds")

# Example 4: Get similar songs with genre and year filters
print("\nExample 4: Similar Songs with Filters")
print("-----------------------------------")
rec_start = time.time()
# Get a valid song_id
song_id = song_df['song_id'].iloc[0]
song_title = song_df[song_df['song_id'] == song_id]['title'].iloc[0]
genre = 0  # First genre cluster
year_range = (2000, 2010)  # Songs from 2000 to 2010
similar_songs = hr.get_similar_songs(song_id, year_range=year_range, genre=genre)
# Merge with song metadata to get titles
similar_songs = pd.merge(similar_songs, song_df[['song_id', 'title', 'artist_name', 'year']].drop_duplicates(), on='song_id')
rec_time = time.time() - rec_start
print(f"\nSimilar songs to '{song_title}' with genre {genre} and years {year_range}")
print(similar_songs[['song_id', 'title', 'artist_name', 'year', 'score']].head())
print(f"Recommendation time: {rec_time:.2f} seconds")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")