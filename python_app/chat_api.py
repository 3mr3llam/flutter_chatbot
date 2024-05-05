import random  # Importing the random module for generating random responses
import json  # Importing the json module for working with JSON data
import torch  # Importing PyTorch for deep learning functionalities
from fastapi import FastAPI  # Importing FastAPI for building the web API
from pydantic import BaseModel  # Importing BaseModel for defining request and response models

from model import ChatNeuralNet  # Importing the ChatNeuralNet class from the model module
from nltk_utils import bag_of_words, tokenize  # Importing utility functions for text processing

app = FastAPI()  # Creating a FastAPI application instance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Checking for GPU availability

# Load intents and model
with open('intents.json', 'r') as json_data:  # Opening and reading intents from a JSON file
    intents = json.load(json_data)

FILE = "data.pth"  # File path for the trained model
data = torch.load(FILE)  # Loading the trained model

# Extracting necessary data from the loaded model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = ChatNeuralNet(input_size, hidden_size, output_size).to(device)  # Initializing the model
model.load_state_dict(model_state)  # Loading the trained model's state
model.eval()  # Setting the model to evaluation mode

bot_name = "Sam"  # Setting the name of the chatbot

class Message(BaseModel):  # Defining a Pydantic model for the request message
    message: str

@app.post("/chat/")  # Defining a POST endpoint for chat interactions
async def chat(message: Message):  # Handler function for processing chat messages
    user_input = message.message  # Extracting the user's message from the request
    sentence = tokenize(user_input)  # Tokenizing the user's message
    X = bag_of_words(sentence, all_words)  # Converting the tokenized message to bag of words
    X = X.reshape(1, X.shape[0])  # Reshaping the input for model compatibility
    X = torch.from_numpy(X).to(device)  # Converting input to PyTorch tensor and moving to device

    output = model(X)  # Forward pass through the model
    _, predicted = torch.max(output, dim=1)  # Getting the predicted class

    tag = tags[predicted.item()]  # Retrieving the predicted tag

    probs = torch.softmax(output, dim=1)  # Calculating softmax probabilities
    prob = probs[0][predicted.item()]  # Getting the probability of the predicted class
    if prob.item() > 0.75:  # Checking if the confidence level is high
        for intent in intents['intents']:  # Iterating through intents
            if tag == intent["tag"]:  # Matching the tag with intents
                return { "response": random.choice(intent['responses']) }  # Returning a random response
    else:
        return { "response": "I do not understand..." }  # Returning a default response if confidence is low
