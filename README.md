# Anime Recommender Backend
Backend for the AI Anime Recommender system. Frontend can be found [here](https://github.com/khave/AnimeRecommenderFrontend)
<br> <br>
The recommender uses a Siamese Bert Network ([link](https://www.sbert.net/index.html)) to do asymmetric semantic search. <br>
The model is a fine-tuned distilbert-dot-tas_b-b256-msmarco ([link](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco)) trained on 163,687 anime reviews (4000 anime in total) scraped from MyAnimeList ([link](https://myanimelist.net/)). <br>
Faiss ([link](https://github.com/facebookresearch/faiss)) is used to speed up the search process. <br>
The search process consists of getting the top N reviews closest to the query and then returning the anime with the most reviews returned.
<br>
The main.py can be run standalone for a console version.