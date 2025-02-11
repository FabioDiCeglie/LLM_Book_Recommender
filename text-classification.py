import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm

books = pd.read_csv("books_cleaned.csv")
books["categories"].value_counts().reset_index()
books["categories"].value_counts().reset_index().query('count > 50')
books[books["categories"] == "Juvenile Fiction"]
books[books["categories"] == "Juvenile Nonfiction"]

category_mapping = {'Fiction' : "Fiction",
 'Juvenile Fiction': "Children's Fiction",
 'Biography & Autobiography': "Nonfiction",
 'History': "Nonfiction",
 'Literary Criticism': "Nonfiction",
 'Philosophy': "Nonfiction",
 'Religion': "Nonfiction",
 'Comics & Graphic Novels': "Fiction",
 'Drama': "Fiction",
 'Juvenile Nonfiction': "Children's Nonfiction",
 'Science': "Nonfiction",
 'Poetry': "Fiction"}

books["simple_categories"] = books["categories"].map(category_mapping)
books[~(books["simple_categories"].isna())]

fiction_categories = ["Fiction", "Nonfiction"]

pipe = pipeline("zero-shot-classification",
                model="facebook/bart-large-mnli",
                device="mps")

sequence = books.loc[books['simple_categories'] == 'Fiction', 'description'].reset_index(drop=True)[0]
pipe(sequence, fiction_categories)

# {'sequence': 'A NOVEL THAT READERS and critics have been eagerly anticipating for over a decade, Gilead is an astonishingly imagined story of remarkable lives. John Ames is a preacher, the son of a preacher and the grandson (both maternal and paternal) of preachers. It’s 1956 in Gilead, Iowa, towards the end of the Reverend Ames’s life, and he is absorbed in recording his family’s story, a legacy for the young son he will never see grow up. Haunted by his grandfather’s presence, John tells of the rift between his grandfather and his father: the elder, an angry visionary who fought for the abolitionist cause, and his son, an ardent pacifist. He is troubled, too, by his prodigal namesake, Jack (John Ames) Boughton, his best friend’s lost son who returns to Gilead searching for forgiveness and redemption. Told in John Ames’s joyous, rambling voice that finds beauty, humour and truth in the smallest of life’s details, Gilead is a song of celebration and acceptance of the best and the worst the world has to offer. At its heart is a tale of the sacred bonds between fathers and sons, pitch-perfect in style and story, set to dazzle critics and readers alike.',
#  'labels': ['Fiction', 'Nonfiction'],
#  'scores': [0.8438265919685364, 0.15617339313030243]}

max_index = np.argmax(pipe(sequence, fiction_categories)["scores"])
max_label = pipe(sequence, fiction_categories)["labels"][max_index]

# Fiction

def generate_predictions(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])
    max_label = predictions["labels"][max_index]
    return max_label

actual_cats = []
predicted_cats = []

for i in tqdm(range(0, 300)):
    sequence = books.loc[books['simple_categories'] == 'Fiction', 'description'].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ['Fiction']

for i in tqdm(range(0, 300)):
    sequence = books.loc[books['simple_categories'] == 'Nonfiction', 'description'].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ['Nonfiction']


predictions_df = pd.DataFrame({'actual_categories': actual_cats, 'predicted_categories': predicted_cats})
predictions_df['correct_prediction'] = (
    np.where(predictions_df['actual_categories'] == predictions_df['predicted_categories'], 1, 0)
)
predictions_df["correct_prediction"].sum() / len(predictions_df)
# 0.77

## START PREDICTION

isbns = []
predicted_cats = []

missing_cats = books.loc[books['simple_categories'].isna(), ['isbn13', 'description']].reset_index(drop=True)

for i in tqdm(range(0, len(missing_cats))):
    sequence = missing_cats['description'][i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    isbns += [missing_cats['isbn13'][i]]

missing_predicted_df = pd.DataFrame({'isbn13': isbns, 'predicted_categories': predicted_cats})

books = pd.merge(books, missing_predicted_df, on='isbn13', how='left')
books['simple_categories'] = np.where(books['simple_categories'].isna(), books['predicted_categories'], books['simple_categories'])
books = books.drop(columns = ["predicted_categories"])
books[books["categories"].str.lower().isin([
    "romance",
    "science fiction",
    "scifi",
    "fantasy",
    "horror",
    "mystery",
    "thriller",
    "comedy",
    "crime",
    "historical"
])]

books.to_csv("books_with_categories.csv", index=False)