# Song Recommendation Engine

This project is a song recommendation engine built using the Million Songs Dataset. The recommendation system uses three types of filtering techniques to provide personalized song recommendations:

- **Popularity-based filtering**
- **Content-based filtering**
- **Collaborative filtering**

## Dataset Information

The Million Songs Dataset consists of two files:

- **triplets_file**: Contains `user_id`, `song_id`, and `listen_count` which represents user interactions with songs.
- **metadata_file**: Contains `song_id`, `title`, `release_year`, and `artist_name`.

## Project Overview

The main objective of this project was to build a recommendation engine that can recommend songs to users based on their listening history and song metadata. The system provides:

- **Popularity-based Recommendations**: Recommends the most popular songs based on how many users have listened to them.
- **Item-based Collaborative Filtering**: Recommends songs similar to the ones a user has already listened to based on item similarity.
- **Content-based Filtering**: Recommends songs based on metadata like title, artist, or genre.

## Features and Functionalities

- **Model Evaluation**: The model’s performance is evaluated using metrics such as **F1 score**, **precision**, **recall**, and **hit rate**.
- **Real-time Recommendations**: The recommendation algorithms provide real-time recommendations for users based on their history.
- **Data Handling**: The data is processed and merged to create an effective dataset for training the recommender systems.
- **Model Training**: The system trains two types of models, popularity-based and item similarity-based models.

## Algorithms Used

- **Popularity Filtering**: Recommends the most popular items (songs) based on user counts.
- **Content-Based Filtering**: Uses the metadata (such as song attributes) to recommend similar songs.
- **Collaborative Filtering**: Builds user-item relationships based on the interaction data and recommends similar items.

## Libraries and Technologies

The following libraries were used to build the recommendation system:

- `pandas`
- `numpy`
- `scikit-learn`
- `Recommenders`

## Metrics for Evaluation

- **F1 Score**: A balance between precision and recall.
- **Precision**: The percentage of recommended songs that are actually relevant to the user.
- **Recall**: The percentage of relevant songs that were recommended.
- **Hit Rate**: Whether at least one relevant song was recommended to the user.

## Code Overview

### 1. **Data Loading and Preprocessing**

```python
# Load the data
triplets_file = 'triplets_file/triplets_file.csv'
song_metadata_file = 'song_data/song_data.csv'

song_df_1 = pd.read_csv(triplets_file, header=None, low_memory=False)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

song_df_2 = pd.read_csv(song_metadata_file, low_memory=False)
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

song_df = song_df.sample(n=50000, random_state=42)  # Taking a larger subset for evaluation
```

### 2. **Model Creation**

```python
# Popularity-based Recommender
pr = Recommenders.popularity_recommender_py()
pr.create(train_data, 'user_id', 'song_id')

# Item-based Similarity Recommender
ir = Recommenders.item_similarity_recommender_py()
ir.create(train_data, 'user_id', 'song_id')
```

### 3. **Evaluation Metrics**

```python
# Calculate metrics such as F1 score, precision, recall, and hit rate for both models
f1_pop, prec_pop, rec_pop, hit_pop = calculate_recommendation_metrics(pr, test_data)
f1_item, prec_item, rec_item, hit_item = calculate_recommendation_metrics(ir, test_data)
```

### 4. **Example Recommendations**

```python
# Popularity-based Recommendations
popular_recommendations = pr.recommend(user_id)

# Item Similarity Recommendations
similar_songs = ir.get_similar_items([song_id])
```

## Running the Project

To run the project, follow these steps:
1. Install the required libraries:

```bash
pip install -r requirements.txt
```
2. Run the recommender script:

```bash
python test_recommender.py
```

## Conclusion
This project demonstrates the development of a song recommendation engine using popular machine learning techniques. The models are evaluated based on relevant metrics to ensure accuracy and performance.
