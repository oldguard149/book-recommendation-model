import sys
import numpy as np
import pandas as pd
from flask import Flask, request, json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import json

df = pd.read_json('./model/book_data.json', orient='table')
countMatrix = pickle.load(open('./model/wordcountmatrix.pkl', 'rb'))
cosine_similarity_score = cosine_similarity(countMatrix, countMatrix)
isbnIndex = pd.Series(data=df.index, index=df.loc[:, 'isbn13'])

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    isbn = request.json['isbn']
    noOfBook = request.json['number']
    try:
        # check is post request isbn is valid
        _ = isbnIndex[isbn]
    except:
        return app.response_class(
            response={"sucess": False, "message": "isbn is invalid"},
            status=400,
            mimetype='application/json'
        )
    isbnList = get_recommendations_with_isbn_index(isbn, noOfBook, isbnIndex).tolist()
    return app.response_class(
        response= json.dumps({'success': True, 'isbnlist': isbnList}),
        status=200,
        mimetype='application/json'
    )


def get_recommendations_with_isbn_index(isbn, numberOfBooksReturn, index=isbnIndex, cosine_sim=cosine_similarity_score):
    # Get similarity score
    idx = index[isbn]
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top scores
    sim_scores = sim_scores[1:(numberOfBooksReturn+1)]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar book
    return df.loc[book_indices, 'isbn13']