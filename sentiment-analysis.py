# we gonna use a technique calling fine-tuning for this.

import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm

books = pd.read_csv("books_with_categories.csv")

classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base",
                      top_k = None,
                      device = "mps")

# Examples
# print(classifier("I love this!"))
# print(books["description"][0])
# print(classifier(books['description']))

# divide the description in different sentences
# print(classifier(books["description"][0].split(".")))

sentences = books['description'][0].split('.')
predictions = classifier(sentences)

# print(predictions[0], sentences[0])

# print(sorted(predictions[0], key=lambda x: x["label"]))

emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

def calculate_max_emotion_scores(predictions):
    per_emotion_scores = { label: [] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

# for i in range(10):
#     isbn.append(books['isbn13'][i])
#     sentences = books['description'][i].split('.')
#     predictions = classifier(sentences)
#     max_scores = calculate_max_emotion_scores(predictions)
#     for label in emotion_labels:
#         emotion_scores[label].append(max_scores[label])

# print(emotion_scores)

for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])