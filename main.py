from fastapi import FastAPI
from typing import Optional, List
from models import DestinationRecommendationAttribute
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model('destination_recommender_model.h5',compile=False)
tv = TfidfVectorizer(max_features=5000)

content_based_data = pd.read_csv('data/content_based_data.csv')
dfs = pd.read_csv('data/dfs.csv')
scaler = StandardScaler()

@app.post('/get-destination-recommendation')
async def get_destination_recommendation(recommendationAttribute: DestinationRecommendationAttribute):

    city = recommendationAttribute.city
    user_preferences = recommendationAttribute.user_preferences


    vectors=tv.fit_transform(content_based_data.preferences).toarray()
    scaler.fit_transform(vectors)

    str_user_preferences = ' '.join(user_preferences)
    user_vector = tv.transform([str_user_preferences]).toarray()

    scaled_item_vecs = scaler.transform(user_vector)

    vms = model.predict(scaled_item_vecs)

    similarities = cosine_similarity(vectors, vms)

    sorted_similarities = np.sort(similarities,axis=0)[::-1]
    sorted_indices = np.argsort(similarities,axis=0)[::-1]

    top_k = 10
    # Retrieve top-k similar places
    top_similar_places = []
    k = 0
    for index in sorted_indices:
        if k >= top_k:
                break
        # print(index)
        # place = df_cleaned[df_cleaned['id']==index[0]]
        place_id = content_based_data.iloc[index].id.values
        place = dfs[dfs['id']==place_id[0]]

        if (place.city == city).all():
            top_similar_places.append(place)
            k += 1

    # top_similar_places
    json_list = []
    for df in top_similar_places:
        json_list.append(df.to_dict(orient='records'))

    json_list

    return {"result": json_list}
