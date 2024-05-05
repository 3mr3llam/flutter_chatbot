from nltk_utils import tokenize, stem, bag_of_words
# import our the intents file
import json
import string
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ChatNeuralNet

# open the json file and get its contents
with open('intents.json') as json_data:
    intents = json.load(json_data)

all_words = [] # list to collect all words
tags = [] # list to collect tags
xy = [] # list to hold the pattern and tegs

# loop through each sentence in our intents patterns
for intent in intents['intents']:

    tag = intent['tag']
    if tag not in tags:
        tags.append(tag)
    
    # tokenize every word
    for pattern in intent['patterns']:
        tokenized_words = tokenize(pattern)
        all_words.extend(tokenized_words)
        
        xy.append((tokenized_words, tag))

# remove the punctuation marks fron the text
# lower and stem words
ignore_words = string.punctuation
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# add to the bag of words
# create a list for the trained data
x_train = []
y_train = [] # hold tags here

# loop over xy array

for (sentence, tag) in xy:
    bag = bag_of_words(sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataSet(Dataset):
    def __init__(self):
        self.number_of_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.number_of_samples


batch_size = 8
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle = True, num_workers=0) # change from 0 to 2 if there is an error

# create the neural network model
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChatNeuralNet(input_size, hidden_size, output_size).to(device)

# create the loss and optimizer
learning_rate = 0.001
num_epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# now let's create the actual training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # call forward
        outputs = model(words)
        # let's calculate the loss
        loss = criterion(outputs, labels)

        #backward and optimizer step
        # we have to empty the gridient first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # every 100 step do the next
    if (epoch+1) % 100 == 0: 
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss loss={loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')