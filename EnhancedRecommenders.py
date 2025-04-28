import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import Recommenders as base_recommenders
from scipy.sparse import csr_matrix

class GenreClassifier:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.label_encoder = LabelEncoder()
        self.genre_mapping = None
        
    def fit(self, song_metadata):
        # Extract features for genre classification
        features = song_metadata[['artist_name', 'title', 'release']].copy()
        
        # Create text features
        features['text_features'] = features['artist_name'].fillna('') + ' ' + \
                                  features['title'].fillna('') + ' ' + \
                                  features['release'].fillna('')
        
        # Vectorize text features
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(features['text_features'])
        
        # Cluster songs into genres
        self.kmeans.fit(X)
        
        # Create genre mapping
        self.genre_mapping = pd.Series(self.kmeans.labels_, index=song_metadata['song_id'])
        
    def get_song_genre(self, song_id):
        if self.genre_mapping is not None and song_id in self.genre_mapping:
            return self.genre_mapping[song_id]
        return None

class ArtistSimilarityRecommender:
    def __init__(self):
        self.artist_to_songs = None
        self.song_to_artist = None
        self.artist_to_idx = None
        self.idx_to_artist = None
        self.artist_song_matrix = None
        
    def create(self, song_metadata):
        # Create artist-song mapping
        self.artist_to_songs = song_metadata.groupby('artist_name')['song_id'].apply(list).to_dict()
        self.song_to_artist = song_metadata.set_index('song_id')['artist_name'].to_dict()
        
        # Create artist and song index mappings
        unique_artists = list(self.artist_to_songs.keys())
        unique_songs = list(song_metadata['song_id'].unique())
        
        self.artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        self.idx_to_artist = {idx: artist for artist, idx in self.artist_to_idx.items()}
        
        # Create sparse artist-song matrix
        rows = []
        cols = []
        data = []
        
        song_to_idx = {song: idx for idx, song in enumerate(unique_songs)}
        
        for artist, songs in self.artist_to_songs.items():
            artist_idx = self.artist_to_idx[artist]
            for song in songs:
                if song in song_to_idx:  # Check if song exists in mapping
                    song_idx = song_to_idx[song]
                    rows.append(artist_idx)
                    cols.append(song_idx)
                    data.append(1)
        
        self.artist_song_matrix = csr_matrix((data, (rows, cols)), 
                                           shape=(len(unique_artists), len(unique_songs)))
        
    def get_similar_artists(self, artist_name, n_recommendations=5):
        if artist_name not in self.artist_to_idx:
            return None
            
        artist_idx = self.artist_to_idx[artist_name]
        artist_vector = self.artist_song_matrix[artist_idx].toarray().flatten()
        
        # Calculate similarity with all other artists
        similarities = []
        for idx in range(len(self.artist_to_idx)):
            if idx != artist_idx:
                other_vector = self.artist_song_matrix[idx].toarray().flatten()
                # Calculate Jaccard similarity
                intersection = np.sum(artist_vector & other_vector)
                union = np.sum(artist_vector | other_vector)
                similarity = intersection / union if union > 0 else 0
                similarities.append((self.idx_to_artist[idx], similarity))
        
        # Sort by similarity and get top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_recommendations]
        
    def get_artist_recommendations(self, song_id, n_recommendations=5):
        if song_id not in self.song_to_artist:
            return None
            
        artist_name = self.song_to_artist[song_id]
        similar_artists = self.get_similar_artists(artist_name, n_recommendations)
        
        if similar_artists is None:
            return None
            
        recommendations = []
        for artist, score in similar_artists:
            artist_songs = self.artist_to_songs[artist]
            recommendations.extend([(song, score) for song in artist_songs])
            
        # Sort by similarity score and get top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

class HybridRecommender:
    def __init__(self, popularity_weight=0.3, item_similarity_weight=0.3, 
                 content_weight=0.2, artist_weight=0.2):
        self.popularity_weight = popularity_weight
        self.item_similarity_weight = item_similarity_weight
        self.content_weight = content_weight
        self.artist_weight = artist_weight
        
        self.popularity_recommender = None
        self.item_similarity_recommender = None
        self.content_recommender = None
        self.artist_recommender = None
        self.genre_classifier = None
        
    def create(self, train_data, song_metadata):
        # Initialize base recommenders
        self.popularity_recommender = base_recommenders.popularity_recommender_py()
        self.popularity_recommender.create(train_data, 'user_id', 'song_id')
        
        self.item_similarity_recommender = base_recommenders.item_similarity_recommender_py()
        self.item_similarity_recommender.create(train_data, 'user_id', 'song_id')
        
        self.content_recommender = base_recommenders.content_based_recommender_py()
        self.content_recommender.create(train_data, 'user_id', 'song_id', song_metadata)
        
        # Initialize artist similarity recommender
        self.artist_recommender = ArtistSimilarityRecommender()
        self.artist_recommender.create(song_metadata)
        
        # Initialize genre classifier
        self.genre_classifier = GenreClassifier()
        self.genre_classifier.fit(song_metadata)
        
    def recommend(self, user_id, n_recommendations=10, year_range=None, genre=None):
        # Get recommendations from each recommender
        pop_recs = self.popularity_recommender.recommend(user_id)
        item_recs = self.item_similarity_recommender.recommend(user_id)
        content_recs = self.content_recommender.recommend(user_id, n_recommendations)
        
        # Combine recommendations with weights
        combined_scores = {}
        
        # Add popularity recommendations
        if pop_recs is not None:
            for _, row in pop_recs.iterrows():
                song_id = row['song_id']
                score = row['score'] * self.popularity_weight
                combined_scores[song_id] = combined_scores.get(song_id, 0) + score
                
        # Add item similarity recommendations
        if item_recs is not None:
            for _, row in item_recs.iterrows():
                song_id = row['song_id']
                score = row['score'] * self.item_similarity_weight
                combined_scores[song_id] = combined_scores.get(song_id, 0) + score
                
        # Add content-based recommendations
        if content_recs is not None:
            for _, row in content_recs.iterrows():
                song_id = row['song_id']
                score = row['similarity_score'] * self.content_weight
                combined_scores[song_id] = combined_scores.get(song_id, 0) + score
                
        # Add artist-based recommendations
        user_songs = self.item_similarity_recommender.get_user_items(user_id)
        for song_id in user_songs:
            artist_recs = self.artist_recommender.get_artist_recommendations(song_id, n_recommendations)
            if artist_recs is not None:
                for rec_song_id, score in artist_recs:
                    combined_scores[rec_song_id] = combined_scores.get(rec_song_id, 0) + \
                                                 score * self.artist_weight
                    
        # Convert to DataFrame
        recommendations = pd.DataFrame({
            'song_id': list(combined_scores.keys()),
            'score': list(combined_scores.values())
        })
        
        # Apply year range filter if specified
        if year_range is not None:
            min_year, max_year = year_range
            song_years = self.content_recommender.song_metadata.set_index('song_id')['year']
            recommendations = recommendations[
                (song_years[recommendations['song_id']] >= min_year) & 
                (song_years[recommendations['song_id']] <= max_year)
            ]
            
        # Apply genre filter if specified
        if genre is not None:
            song_genres = self.genre_classifier.genre_mapping
            recommendations = recommendations[
                song_genres[recommendations['song_id']] == genre
            ]
            
        # Sort by combined score and get top recommendations
        recommendations = recommendations.sort_values('score', ascending=False)
        return recommendations.head(n_recommendations)
        
    def get_similar_songs(self, song_id, n_recommendations=10, year_range=None, genre=None):
        # Get content-based similar songs
        content_similar = self.content_recommender.get_song_similarity(song_id, n_recommendations)
        
        # Get artist-based similar songs
        artist_similar = self.artist_recommender.get_artist_recommendations(song_id, n_recommendations)
        
        # Combine recommendations
        combined_scores = {}
        
        # Add content-based recommendations
        if content_similar is not None:
            for _, row in content_similar.iterrows():
                similar_song_id = row['song_id']
                score = row['similarity_score'] * self.content_weight
                combined_scores[similar_song_id] = combined_scores.get(similar_song_id, 0) + score
                
        # Add artist-based recommendations
        if artist_similar is not None:
            for similar_song_id, score in artist_similar:
                combined_scores[similar_song_id] = combined_scores.get(similar_song_id, 0) + \
                                                 score * self.artist_weight
                                                 
        # Convert to DataFrame
        recommendations = pd.DataFrame({
            'song_id': list(combined_scores.keys()),
            'score': list(combined_scores.values())
        })
        
        # Apply year range filter if specified
        if year_range is not None:
            min_year, max_year = year_range
            song_years = self.content_recommender.song_metadata.set_index('song_id')['year']
            recommendations = recommendations[
                (song_years[recommendations['song_id']] >= min_year) & 
                (song_years[recommendations['song_id']] <= max_year)
            ]
            
        # Apply genre filter if specified
        if genre is not None:
            song_genres = self.genre_classifier.genre_mapping
            recommendations = recommendations[
                song_genres[recommendations['song_id']] == genre
            ]
            
        # Sort by combined score and get top recommendations
        recommendations = recommendations.sort_values('score', ascending=False)
        return recommendations.head(n_recommendations) 