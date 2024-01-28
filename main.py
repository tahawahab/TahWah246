import re
import spacy
import uvicorn
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
from model_training import preprocess_text
#FastAPI Application Setup and Text Classification Route

# Import the FastAPI library
app = FastAPI()

# Loading the saved model
model = load('logistic_regression_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')
label_encoder = load('label_encoder.joblib')

# Define a data model for the input query
class Query(BaseModel):
    text: str

# Define a FastAPI route for classifying text
@app.post("/classify/")
def classify(query: Query):
    # Apply the preprocessing to the incoming query text
    processed_text = preprocess_text(query.text)
    
    # Vectorize the preprocessed text
    vectorized_text = vectorizer.transform([processed_text])
    
    # Predict the label
    prediction = model.predict(vectorized_text)

    # Log or print the prediction to verify its format
    print("Prediction (numeric):", prediction)

    # Ensure the prediction is in the correct format for inverse_transform
    if prediction.size > 0:
        label = label_encoder.inverse_transform(prediction)[0]
    else:
        label = "Unknown"

    # Return the input query text and the predicted label as a response
    return {"query": query.text, "prediction": label}