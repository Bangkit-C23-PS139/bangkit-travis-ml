from fastapi import FastAPI
from typing import Optional, List
from models import DestinationRecommendationAttribute
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import json
from supabase import Client, create_client
from dotenv import load_dotenv
from os import environ

app = FastAPI()

model = tf.keras.models.load_model('destination_recommender_model.h5',compile=False)
model2 = tf.keras.models.load_model('restaurant_recommender_model.h5',compile=False)
tv = TfidfVectorizer(max_features=5000)
tv2 = TfidfVectorizer(max_features=5000)

content_based_data = pd.read_csv('data/content_based_data.csv')
resto_content_based_data = pd.read_csv('data/resto_content_based_data.csv')
dfs = pd.read_csv('data/dfs.csv')
scaler = StandardScaler()
scaler2 = StandardScaler()

def hotel_recommender(filtered_hotel_df, city, top_k=3):
    hotel_by_city_df = filtered_hotel_df[filtered_hotel_df['city'] == city]
    random_rows = hotel_by_city_df.sample(n=top_k)
    return random_rows

def get_filtered_hotel_df():

    hotel_df = dfs[dfs.keyword_category.apply(lambda x: isinstance(x,str) and 'penginapan' in x)]
    hotel_df.drop(['Unnamed: 0'],axis=1,inplace=True)
        
    filtered_hotel_df = hotel_df[hotel_df['rating'] >= 4.4]
    filtered_hotel_df = filtered_hotel_df[filtered_hotel_df['total_review'] > 30]

    return filtered_hotel_df


def get_recommendation_cosine(city: str, user_preferences: List[str]):
    vectors=tv.fit_transform(content_based_data.preferences).toarray()

    str_user_preferences = ' '.join(user_preferences)

    user_vector = tv.transform([str_user_preferences]).toarray()

    # Calculate similarity scores between user preferences and content data
    similarity_scores =tf.keras.losses.cosine_similarity(user_vector, vectors)

    sorted_indices = tf.argsort(similarity_scores)

    k = 0
    top_k = 10
    top_similar_places = []
    for index in sorted_indices:

        if k >= top_k:
            break
        
        place_id = content_based_data.iloc[index.numpy()].id
        place = dfs[dfs['id']==place_id]
        if (place.city == city).all():
            top_similar_places.append(place)
            k += 1

    return top_similar_places

def get_recommendation_rest_cosine(city: str, user_preferences: List[str]):
    vectors=tv.fit_transform(resto_content_based_data.preferences).toarray()

    str_user_preferences = ' '.join(user_preferences)

    user_vector = tv.transform([str_user_preferences]).toarray()

    # Calculate similarity scores between user preferences and content data
    similarity_scores =tf.keras.losses.cosine_similarity(user_vector, vectors)

    sorted_indices = tf.argsort(similarity_scores)

    k = 0
    top_k = 10
    top_similar_places = []
    for index in sorted_indices:

        if k >= top_k:
            break
        
        place_id = resto_content_based_data.iloc[index.numpy()].id
        place = dfs[dfs['id']==place_id]
        if (place.city == city).all():
            top_similar_places.append(place)
            k += 1

    return top_similar_places

def combine_dataframes(df_list):
    combined_df = pd.DataFrame()
    for df in df_list:
        combined_df = combined_df.append(df, ignore_index=True)
    return combined_df

def df_to_json(lst_row_df):
    json_res = []
    for df in lst_row_df:
        json_res.append(df.to_dict())

@app.post('/get-destination-recommendation')
async def get_destination_recommendation(recommendationAttribute: DestinationRecommendationAttribute):

    city = recommendationAttribute.city
    user_dest_preferences = recommendationAttribute.user_destination_preferences
    user_restaurant_preferences = recommendationAttribute.user_restaurant_preferences

    # recommend destination
    vectors=tv.fit_transform(content_based_data.preferences).toarray()
    scaler.fit_transform(vectors)

    str_user_dst_preferences = ' '.join(user_dest_preferences)
    user_dest_vector = tv.transform([str_user_dst_preferences]).toarray()

    scaled_item_dest_vecs = scaler.transform(user_dest_vector)

    vms = model.predict(scaled_item_dest_vecs)

    similarities = cosine_similarity(vectors, vms)

    sorted_similarities = np.sort(similarities,axis=0)[::-1]
    sorted_indices = np.argsort(similarities,axis=0)[::-1]

    top_k = 20
    # Retrieve top-k similar places
    top_similar_places_model = []
    k = 0
    for index in sorted_indices:
        if k >= top_k:
                break
        place_id = content_based_data.iloc[index].id.values

        place = dfs[dfs['id']==place_id[0]]

        if (place.city == city).all():
            top_similar_places_model.append(place)
            k += 1

    top_similar_places_cosine = get_recommendation_cosine(city, user_dest_preferences)

    recom_cosine = pd.DataFrame()
    recom_model = pd.DataFrame()
    
    recom_cosine = combine_dataframes(top_similar_places_cosine)
    recom_cosine.drop(['Unnamed: 0'],axis=1,inplace=True)
    url_cosine_lst = recom_cosine.map_url.tolist()

    
    recom_model = combine_dataframes(top_similar_places_model)
    recom_model.drop(['Unnamed: 0'],axis=1,inplace=True)
    url_model_lst = recom_model.map_url.tolist()


    duplicates = [url for url in url_model_lst if url in url_cosine_lst]

    recom_model = recom_model[~recom_model['map_url'].isin(duplicates)]
    
    recom_dest = pd.concat([recom_cosine.sample(5), recom_model.sample(5)], ignore_index=True)

    recom_dest_lst = [i for _,i in recom_dest.iterrows()]

    json_recom_dest = []
    for df in recom_dest_lst:
        json_recom_dest.append(df.to_dict())

    # recommend hotel
    filtered_hotel_df = get_filtered_hotel_df()

    hotel_recommendation_df = hotel_recommender(filtered_hotel_df, city)

    recom_hotel_lst = [i for _,i in hotel_recommendation_df.iterrows()]

    json_hotel_dest = []
    for df in recom_hotel_lst:
        json_hotel_dest.append(df.to_dict())


    # recommend tempat makan    
    vectors2=tv2.fit_transform(resto_content_based_data.preferences).toarray()
    scaler2.fit_transform(vectors2)

    str_user_restaurant_preferences = ' '.join(user_restaurant_preferences)
    user_rest_vector = tv2.transform([str_user_restaurant_preferences]).toarray()

    scaled_item_rest_vecs = scaler2.transform(user_rest_vector)

    vms2 = model2.predict(scaled_item_rest_vecs)

    similarities2 = cosine_similarity(vectors2, vms2)

    sorted_similarities2 = np.sort(similarities2,axis=0)[::-1]
    sorted_indices2 = np.argsort(similarities2,axis=0)[::-1]

    top_k = 15
    # Retrieve top-k similar places
    top_similar_restaurant_model = []
    k = 0
    for index in sorted_indices2:
        if k >= top_k:
                break
        place_id = resto_content_based_data.iloc[index].name.values

        place = dfs[dfs['name']==place_id[0]]

        if (place.city == city).all():
            top_similar_restaurant_model.append(place)
            k += 1

    top_similar_restaurant_cosine = get_recommendation_rest_cosine(city, user_restaurant_preferences)

    recom_rest_cosine = pd.DataFrame()
    recom_rest_model = pd.DataFrame()
    
    recom_rest_cosine = combine_dataframes(top_similar_restaurant_cosine)
    recom_rest_cosine.drop(['Unnamed: 0'],axis=1,inplace=True)
    url_cosine_rest_lst = recom_rest_cosine.map_url.tolist()

    recom_rest_model = combine_dataframes(top_similar_restaurant_model)
    recom_rest_model.drop(['Unnamed: 0'],axis=1,inplace=True)
    url_model_rest_lst = recom_rest_model.map_url.tolist()


    duplicates = [url for url in url_model_rest_lst if url in url_cosine_rest_lst]

    recom_rest_model = recom_rest_model[~recom_rest_model['map_url'].isin(duplicates)]
    
    recom_rest = pd.concat([recom_rest_cosine.sample(3), recom_rest_model.sample(2)], ignore_index=True)

    recom_rest_lst = [i for _,i in recom_rest.iterrows()]

    json_recom_rest = []
    for df in recom_rest_lst:
        json_recom_rest.append(df.to_dict())

    return {"result": 
        {
            "destination_recommendation": json_recom_dest,
            "hotel_recommendation": json_hotel_dest,
            "restaurant_recommendation": json_recom_rest
        }
    }