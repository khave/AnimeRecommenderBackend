from main import load_model, load_dataset, load_index, load_cross_encoder, retreive_and_rerank, search, download_files
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

download_files()

print("Loading model...")
model = load_model()
cross_encoder = load_cross_encoder()

print("Loading dataset...")
df = load_dataset()

print("Loading faiss indexes...")
index = load_index()

print(f"Amount of reviews: {len(df)}")


@app.route('/search', methods=['POST'])
@cross_origin()
def predict():
    """
    Post a query to the server and get the results.
    """
    query = request.form['query']
    print(f"Received query: {query}")
    raw_results, results = search(query=query, df=df, top_n=150, index=index, model=model)
    # Remove the 'Review' column from the results
    results = results.drop(columns=['Review'])
    #retrieval_results = retreive_and_rerank(cross_encoder, query, raw_results, top_k=50)

    # Return both the results and the retrieval results in one json object. Take only the top 18 results
    # return {'results': results.head(18).to_dict(orient='records'), 'retrieval_results': retrieval_results.head(18).to_dict(orient='records')}
    return {'results': results.head(18).to_dict(orient='records')}
    
 

if __name__ == '__main__':
    port = 5000
    print(f"Server running on port {port}")
    app.run(host='localhost', port=port)
