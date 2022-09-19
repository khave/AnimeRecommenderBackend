import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import time
import gdown
import os
import zipfile


MODEL_NAME = 'data/anirec-model-v1' # search-model-v3-L-GPL-L or v4
DATASET_NAME = 'data/all_reviews_full_plus.csv'
# Best models are probably search-model-v3-L-GPL or search-model-v3-L
INDEX_FILE_NAME = 'data/anime-reviews-anirec-model-v1.index'
CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'


# This search is better for themes since it doesn't rank individual reviews, but ranks the entire review dataset per anime as a whole.


def search(query, df, top_n, index, model):
    """
    Searches the reviews for the given query.

    @param query: The query to search for.
    @param df: The dataframe with the reviews.
    @param top_n: The number of results to return.
    @param index: The faiss index to use.
    @param model: The sentence transformer model to use.

    @return:
        raw_results: The raw results from the faiss index as a dataframe. Will contain top_n entries
        results: The results from the faiss index, but with duplicates dropped and sorted by the count of same anime titles. Will **NOT** contain top_n entries.
    """
    t = time.time()
    query_vector = model.encode([query])
    #query_vector = encode(model, query)
    top_k = index.search(query_vector, top_n) # We can't just top_n = len(df) since then we will just return all the results. 
    # We could use the old search with similarities but that is slower.
    top_k_ids = top_k[1].tolist()[0]
    # Get the anime titles for the top_k_ids without using numpy
    
    # Create a list of all the unique top_k_ids
    top_k_ids = list(set(top_k_ids))
    #top_k_ids_np = list(np.unique(top_k_ids))

    # Assert whether the top_k_ids are the same as the top_k_ids_np
    #assert top_k_ids == top_k_ids_np

    # These are the raw results i.e. the reviews that have the highest similarity to the query.
    raw_results = df.iloc[top_k_ids].copy()

    results = df.iloc[top_k_ids].copy()

    results['count'] = results.groupby('Anime Title', sort=False).cumcount() + 1

    results = results.sort_values(by='count', ascending=False)

    results.drop_duplicates(subset='Anime Title', keep='first', inplace=True)

    # Set all NaN values in Anime URL to empty string
    results['Image URL'] = results['Image URL'].fillna('')

    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    return raw_results.iloc[:, :4], results.iloc[:, :5]


def load_model():
    return SentenceTransformer(MODEL_NAME) # Previosuly all-mpnet-base-v2


def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_NAME)
    

def retreive_and_rerank(cross_encoder, query, results, top_k=100):
    """
    Retreives the top k results and reranks them using the cross_encoder.
    """
    t = time.time()
    # Take only the top top_k results and rerank those
    results = results.head(top_k)
    # Create a list of query and reviews to be used for the cross-encoder
    cross_inp = [[query, review] for anime_title, review in results['Review'].items()]
    cross_scores = cross_encoder.predict(cross_inp)
    results.insert(3, 'Cross Score', cross_scores)
    #results['Cross Score'] = cross_scores
    results = results.sort_values(by='Cross Score', ascending=False)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    return results
 

def load_dataset():
    """
    Loads the dataset from the given path.
    """   
    df = pd.read_csv(DATASET_NAME)

    return df


def load_index():
    return faiss.read_index(INDEX_FILE_NAME)


def main():
    print("Loading model...")
    model = load_model()
    cross_encoder = load_cross_encoder()

    print("Loading dataset...")
    df = load_dataset()

    print("Loading faiss indexes...")
    index = load_index()

    print(f"Amount of reviews: {len(df)}")

    # Keep asking the user for a query until they enter 'quit'
    while True:
        query = input("Enter a query (q or quit to exit): ")
        if query.lower() == 'quit' or query.lower() == 'q' or query.lower() == 'exit':
            print("Goodbye!")
            break
        raw_results, results = search(query=query, df=df, top_n=1500, index=index, model=model) # We have around 26,000 reviews in the dataset, so we get the top 1000 reviews for each query.
        print(f"Results:\n{results.head(18)}\n")
        print(f"Raw results:\n{raw_results.head(18)}\n")
        print("Retreiving and reranking...")
        retrieval_results = retreive_and_rerank(cross_encoder, query, raw_results, top_k=50)
        print(f"Retrieval Results:\n{retrieval_results.head(8)}\n")
        # # Print the reviews
        # count = 0; 
        # for anime_title, review in retrieval_results['Review'].items():
        #     print(f"{anime_title}\n{review}\n")
        #     count += 1
        #     if count == 2:
        #         count = 0
        #         break
        # print("\n")


if __name__ == '__main__':
    main()
