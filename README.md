# Song Recommendation Engine

This project is a song recommendation engine built using the Million Songs Dataset. The recommendation system uses multiple filtering techniques to provide personalized song recommendations:

- Popularity-based filtering
- Content-based filtering
- Collaborative filtering
- Artist-based filtering
- Hybrid recommendations

## Dataset Information
The Million Songs Dataset consists of two files:
- `triplets_file`: Contains `user_id`, `song_id`, and `listen_count` which represents user interactions with songs.
- `metadata_file`: Contains `song_id`, `title`, `release_year`, and `artist_name`.

(The dataset files are uploaded on Drive due to their larger size, [Drive Link](https://drive.google.com/drive/folders/1BxdFjDC774GiUqZcU01foMqZAgIR1XcS?usp=sharing))

## Project Overview
The main objective of this project was to build a recommendation engine that can recommend songs to users based on their listening history and song metadata. The system provides:

- **Popularity-based Recommendations**: Recommends the most popular songs based on how many users have listened to them.
- **Item-based Collaborative Filtering**: Recommends songs similar to the ones a user has already listened to based on item similarity.
- **Content-based Filtering**: Recommends songs based on metadata like title, artist, or genre.
- **Artist-based Recommendations**: Recommends songs based on artist similarity using Jaccard similarity.
- **Hybrid Recommendations**: Combines multiple recommendation strategies for better results.

## Features and Functionalities
- **Model Evaluation**: The model's performance is evaluated using metrics such as F1 score, precision, recall, and hit rate.
- **Real-time Recommendations**: The recommendation algorithms provide real-time recommendations for users based on their history.
- **Data Handling**: The data is processed and merged to create an effective dataset for training the recommender systems.
- **Model Training**: The system trains multiple types of models:
  - Popularity-based model
  - Item similarity-based model
  - Content-based model
  - Artist similarity model
  - Hybrid model
- **Memory Optimization**: Implements sparse matrices and efficient data structures for handling large datasets.
- **Caching**: Uses LRU caching for faster recommendations.

## Algorithms Used
- **Popularity Filtering**: Recommends the most popular items (songs) based on user counts.
- **Content-Based Filtering**: Uses the metadata (such as song attributes) to recommend similar songs.
- **Collaborative Filtering**: Builds user-item relationships based on the interaction data and recommends similar items.
- **Artist Similarity**: Uses Jaccard similarity to find similar artists and their songs.
- **Hybrid Approach**: Combines multiple recommendation strategies with weighted scores.

## Libraries and Technologies
The following libraries were used to build the recommendation system:
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `joblib`

## Project Structure
```
Song-Recommendation-System/
├── src/
│   ├── Recommenders.py          # Core recommendation algorithms
│   ├── EnhancedRecommenders.py  # Advanced recommendation features
│   └── test_recommender.py      # Testing and evaluation script
├── notebooks/
│   └── Million Songs Data - Recommendation Engine.ipynb
├── data/
│   ├── song_data/              # Song metadata
│   └── triplets_file/          # User-song interactions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation Requirements
1. Python 3.7 or higher
2. Required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/labishbardiya/Song-Recommendation-System.git
   cd Song-Recommendation-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the recommender script:
   ```bash
   python test_recommender.py
   ```

## Memory Optimization
The project implements several memory optimization techniques:
- Sparse matrices for user-song interactions
- Efficient data structures for artist similarity
- Subset of data for evaluation
- Caching mechanisms for faster recommendations

## Metrics for Evaluation
- **F1 Score**: A balance between precision and recall.
- **Precision**: The percentage of recommended songs that are actually relevant to the user.
- **Recall**: The percentage of relevant songs that were recommended.
- **Hit Rate**: Whether at least one relevant song was recommended to the user.

## Code Examples
1. **Data Loading and Preprocessing**
```python
# Load the data
triplets_file = 'triplets_file/triplets_file.csv'
song_metadata_file = 'song_data/song_data.csv'

song_df_1 = pd.read_csv(triplets_file, header=None, low_memory=False)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

song_df_2 = pd.read_csv(song_metadata_file, low_memory=False)
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
```

2. **Model Creation**
```python
# Popularity-based Recommender
pr = Recommenders.popularity_recommender_py()
pr.create(train_data, 'user_id', 'song_id')

# Item-based Similarity Recommender
ir = Recommenders.item_similarity_recommender_py()
ir.create(train_data, 'user_id', 'song_id')

# Content-based Recommender
cr = Recommenders.content_based_recommender_py()
cr.create(train_data, 'user_id', 'song_id', song_metadata)

# Hybrid Recommender
hr = EnhancedRecommenders.HybridRecommender()
hr.create(train_data, song_metadata)
```

3. **Getting Recommendations**
```python
# Popularity-based Recommendations
popular_recommendations = pr.recommend(user_id)

# Item Similarity Recommendations
similar_songs = ir.get_similar_items([song_id])

# Content-based Recommendations
content_recommendations = cr.recommend(user_id)

# Hybrid Recommendations
hybrid_recommendations = hr.recommend(user_id)
```

## Other Contributors

- [Himanshu Garg](https://www.github.com/himanshu-garg10)
- [Rakshika Sharma](https://www.github.com/rakshika1)

## Conclusion
This project demonstrates the development of a song recommendation engine using popular machine learning techniques. The models are evaluated based on relevant metrics to ensure accuracy and performance.
