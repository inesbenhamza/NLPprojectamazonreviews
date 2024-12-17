
#importing the necessary librabries

import pandas as pd
import spacy
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import re 
import transformers 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

import kagglehub

# Download latest version
path = kagglehub.dataset_download("mexwell/amazon-reviews-multi")

print("Path to dataset files:", path)

test.head()




#PREPROCESSING

# Concatenate the DataFrames along rows 
merged_df = pd.concat([test, val, train], axis=0)

merged_df = merged_df.reset_index(drop=True)

# Display the result
print(merged_df.head())
print("Total rows:", len(merged_df))



unique=merged_df["language"].unique()
print(unique)

#unique language values : ['de' 'en' 'es' 'fr' 'ja' 'zh']



#keeping only roman languages (beside german)

# Drop rows where the 'language' column is 'ja','zh' or 'de'
df = merged_df[~merged_df['language'].isin(['ja', 'zh', 'de'])]

# Display the result
print(df['language'].unique())  # Check remaining languages
#'en' 'es' 'fr'

print("Total rows after filtering:", len(df))

#Total rows after filtering: 840000

df.shape
#840 000 reviews (rows) and 9 columns 
len(df)
#840 000 values


# review rating is a score from 1 to 5 

df ['review_body'].values[0]




# displaying the number of categories 

# Group by 'product_category' and count the number of reviews per category

df_reviews_per_category = df.groupby('product_category').size()
print(df_reviews_per_category)

df_reviews_per_category.count()





####################


# Count of reviews by stars for each language
df_reviews_per_language = df.groupby(['language', 'stars']).size().unstack()

# Plotting
df_reviews_per_language.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    title='Count of Reviews by Stars per Language'
)
plt.xlabel('Language')
plt.ylabel('Count of Reviews')
plt.legend(title='Stars')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Plotting count of reviews per star
df['stars'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews per Star', figsize=(10, 5))
plt.xlabel('Stars')
plt.ylabel('Count of Reviews')
plt.show()



#there are an equal number of reviews per stars. So as many 1, 2, 3, 4 and 5 stars 

#optional balanced sampling of the reviews for speeding up the task 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def balance_dataset(df, max_samples_per_group=5000):
    #we created a column sentiment

    df['sentiment'] = pd.cut(df['stars'], 
                              bins=[0, 2, 3, np.inf], 
                              labels=['Negative', 'Neutral', 'Positive'])

    #	•	negative: Star ratings from 0 to 2 (e.g., 1, 2).
	#   •	neutral: Star rating of 3
	#   •	positive: Star ratings greater than 3 (e.g., 4, 5).
    

    balanced_samples = []
    

    languages = df['language'].unique()
    star_ratings = df['stars'].unique()
    
    for lang in languages:
        for stars in star_ratings:
            # Filter by language and star rating
            lang_stars_df = df[(df['language'] == lang) & (df['stars'] == stars)]
            
            # Sample up to max_samples_per_group
            if len(lang_stars_df) > max_samples_per_group:
                sampled = lang_stars_df.sample(n=max_samples_per_group, random_state=42)
            else:
                sampled = lang_stars_df
            
            balanced_samples.append(sampled)
    
   
    balanced_df = pd.concat(balanced_samples)
    
    return balanced_df

def advanced_sampling(df, total_samples=400000):
    
    balanced_df = balance_dataset(df)
    
    
    if len(balanced_df) > total_samples:
       
        _, sampled_df = train_test_split(
            balanced_df, 
            train_size=total_samples/len(balanced_df),
            stratify=balanced_df[['language', 'stars']]
        )
    else:
        sampled_df = balanced_df
    
    return sampled_df

df = advanced_sampling(df, total_samples=400000)

# Verify balance
print("Language Distribution:")
print(df['language'].value_counts())

print("\nStars Distribution:")
print(df['stars'].value_counts())

print("\nLanguage-Stars Distribution:")
print(df.groupby(['language', 'stars']).size())


# Define stopwords for each language
stopwords_dict = {
    'en': set(stopwords.words('english')),
    'fr': set(stopwords.words('french')),
    'es': set(stopwords.words('spanish'))
}





#prepreocessing 
def preprocess_text(text):
    # Remove special characters and lowercase the reviews
    text = re.sub(r'[^\w\s]', '', text.lower(), flags=re.UNICODE)
    # Tokenizing the text
    tokens = word_tokenize(text)
     # Removing stopwords
    tokens = [word for word in tokens if word not in stopwords_dict]
    return tokens 

df['cleaned_reviews'] = df['review_body'].apply(preprocess_text)


print(df['cleaned_reviews'].iloc[0]) #just to check 




##################





#SENTIMENT ANALYSIS 



#VADER- BAG OF WORDS APPROACH 
# Installing NLTK and download the VADER lexicon
import nltk

#english tag 
nltk.download('averaged_perceptron_tagger_eng')

from nltk import pos_tag

tokens = ['this', 'is', 'a', 'test']
print(pos_tag(tokens))

################# correction des cleaned reviews 



df_english = df[df['language'] == "en"]

df_french = df[df['language'] == "fr"]

df_spanish = df[df['language'] == "es"]


df_french['cleaned_reviews'] = df_french['review_body'].apply(preprocess_text)
df_english['cleaned_reviews'] = df_english['review_body'].apply(preprocess_text)
df_spanish ['cleaned_reviews'] = df_spanish['review_body'].apply(preprocess_text)

print(df_french['cleaned_reviews'].iloc[0]) 

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


import nltk
nltk.download('vader_lexicon')

##################################VADEEEEEEERRRR############################

# Initializing the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# creating a function to get sentiment scores
def get_sentiment_scores(text):

    sentiment_scores = sia.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    

    return {
        'text': text,
        'pos_score': sentiment_scores['pos'],
        'neg_score': sentiment_scores['neg'],
        'neu_score': sentiment_scores['neu'],
        'compound_score': sentiment_scores['compound'],
        'vader_sentiment': sentiment  # Renamed key to avoid conflicts
    }


df_english['sentiment_analysis'] = df_english['review_body'].apply(get_sentiment_scores)

#Extracting specific columns from the sentiment analysis
df_english['vader_sentiment'] = df_english['sentiment_analysis'].apply(lambda x: x['vader_sentiment'])
df_english['vader_compound_score'] = df_english['sentiment_analysis'].apply(lambda x: x['compound_score'])

#Visualization
vader_sentiment_counts = df_english['vader_sentiment'].value_counts()
print("VADER Sentiment Distribution:")
print(vader_sentiment_counts)



############# matrix classification for vader against true ratings
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df_english['sentiment'] = df_english['sentiment'].str.lower() 
df_english['vader_sentiment'] = df_english['vader_sentiment'].str.lower()  


y_true = df_english['sentiment'] 
y_pred = df_english['vader_sentiment'] 


labels = ['negative', 'neutral', 'positive']

cm = confusion_matrix(y_true, y_pred, labels=labels)


print("Confusion Matrix:")
print(cm)


print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Real Sentiment vs. VADER Sentiment")
plt.show()





### trasnformers 
import torch
import tensorflow as tf
from transformers import pipeline

# Initializing multilingual sentiment analysis models 
models = {
    "nlptown/bert-base-multilingual-uncased-sentiment": pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        use_fast=False
    ),
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        use_fast=False
    ),
    "LiYuan/amazon-review-sentiment-analysis": pipeline(
        "sentiment-analysis",
        model="LiYuan/amazon-review-sentiment-analysis",
        use_fast=False
    ),
}



################ TEST / example of sentence to see how they perform

english_sentence = "This product is very awful and too expensive!"
french_sentence = "Ce produit est vraiment affreux et trop cher!"
spanish_sentence = "Este producto es muy horrible y demasiado caro!"

# Analyze sentiment for each sentence with each model
for model_name, sentiment_pipeline in models.items():
    print(f"\nModel: {model_name}")
    
    # English sentence
    english_result = sentiment_pipeline(english_sentence)
    print(f"English Sentence: {english_sentence}")
    print(f"Sentiment: {english_result[0]['label']}, Score: {english_result[0]['score']}")
    
    # French sentence
    french_result = sentiment_pipeline(french_sentence)
    print(f"French Sentence: {french_sentence}")
    print(f"Sentiment: {french_result[0]['label']}, Score: {french_result[0]['score']}")
    
    # Spanish sentence
    spanish_result = sentiment_pipeline(spanish_sentence)
    print(f"Spanish Sentence: {spanish_sentence}")
    print(f"Sentiment: {spanish_result[0]['label']}, Score: {spanish_result[0]['score']}")


# dropping reviews with more than 512 tokens using AUtotokenizer for english 

from transformers import AutoTokenizer


# Initialize tokenizer for english review 
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


df_english['token_count'] = df_english['review_body'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))
print(df_english['token_count'].describe())



df_english = df_english[df_english['token_count'] <= 512].copy()


df_english = df_english.drop(columns=['token_count'])





# Initialize tokenizer for french review
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Count tokens in each review
df_french['token_count'] = df_french['review_body'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))
print(df_french['token_count'].describe())


df_french = df_french[df_french['token_count'] <= 512].copy()

df_french = df_french.drop(columns=['token_count'])




# Initialize tokenizer for spanish review
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


df_spanish['token_count'] = df_spanish['review_body'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))
print(df_spanish['token_count'].describe())


df_spanish = df_spanish[df_spanish['token_count'] <= 512].copy()

df_spanish = df_spanish.drop(columns=['token_count'])





#########################nlptown/bert-base-multilingual-uncased-sentiment


def analyze_sentiments(df, text_column, model_key, models):

    sentiment_model = models[model_key]
    
   
    sentiment_results = df[text_column].apply(lambda x: sentiment_model(x)[0])  # Apply model on each row
    

    df[f'{model_key}_label'] = sentiment_results.apply(lambda x: x['label'])
    df[f'{model_key}_score'] = sentiment_results.apply(lambda x: x['score'])
    
    return df

`
if __name__ == "__main__":
    import torch
    from transformers import pipeline


    if torch.backends.mps.is_available():
        device = 0  # For MPS, device is 0
        print("Using MPS for hardware acceleration.")
    else:
        device = -1  # Default to CPU if MPS is not available
        print("MPS not available. Using CPU.")
    

    text_column = 'review_body'
    

    models = {
        "nlptown/bert-base-multilingual-uncased-sentiment": pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            use_fast=True,
            device=device
        )
    }
    

    model_key = "nlptown/bert-base-multilingual-uncased-sentiment"
    df_english = analyze_sentiments(df_english, text_column=text_column, model_key=model_key, models=models)
    df_french = analyze_sentiments(df_french, text_column=text_column, model_key=model_key, models=models)
    df_spanish = analyze_sentiments(df_spanish, text_column=text_column, model_key=model_key, models=models)






################evaluating nlp bert 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Map BERT scores to sentiment categories
def map_bert_to_sentiment(star_label):
    
    star = int(star_label.split()[0])
    if star in [1, 2]:
        return 'negative'
    elif star == 3:
        return 'neutral'
    elif star in [4, 5]:
        return 'positive'


for df in [df_english, df_spanish, df_french]:
    df['mapped_sentiment'] = df['nlptown/bert-base-multilingual-uncased-sentiment_label'].apply(map_bert_to_sentiment)
    # Normalize case for consistency
    df['sentiment'] = df['sentiment'].str.lower()
    df['mapped_sentiment'] = df['mapped_sentiment'].str.lower()


labels = ['negative', 'neutral', 'positive']


def plot_confusion_matrix(df, true_column, pred_column, labels, title):
    cm = confusion_matrix(df[true_column], df[pred_column], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


for language, df in [('English', df_english), ('Spanish', df_spanish), ('French', df_french)]:
    print(f"\nConfusion Matrix and Metrics for {language} Reviews:")
    print("Unique values in true labels (sentiment):", df['sentiment'].unique())
    print("Unique values in predicted labels (mapped_sentiment):", df['mapped_sentiment'].unique())
    

    cm = confusion_matrix(df['sentiment'], df['mapped_sentiment'], labels=labels)
    print(f"\nConfusion Matrix for {language} Reviews:")
    print(cm)
    

    plot_confusion_matrix(
        df,
        true_column='sentiment',
        pred_column='mapped_sentiment',
        labels=labels,
        title=f"Confusion Matrix for {language} Reviews"
    )
    

    print(f"\nClassification Report for {language} Reviews:")
    print(classification_report(
        df['sentiment'], 
        df['mapped_sentiment'], 
        target_names=labels, 
        zero_division=0
    ))
    

    
    accuracy_sklearn = accuracy_score(df['sentiment'], df['mapped_sentiment'])
    print(f"Accuracy (from sklearn) for {language} Reviews: {accuracy_sklearn:.2f}")

##########sentiment analysis unsing LiYuan/amazon-review-sentiment-analysis



def analyze_sentiments(df, text_column, model_key, models):
    """
    Perform sentiment analysis on a DataFrame column and append the results.
    """

    sentiment_model = models[model_key]
    

    sentiment_results = df[text_column].apply(lambda x: sentiment_model(x)[0])  # Apply model on each row
    

    df[f'{model_key}_label'] = sentiment_results.apply(lambda x: x['label'])
    df[f'{model_key}_score'] = sentiment_results.apply(lambda x: x['score'])
    
    return df


if __name__ == "__main__":
    import torch
    from transformers import pipeline


    if torch.backends.mps.is_available():
        device = 0  # For MPS, device is 0
        print("Using MPS for hardware acceleration.")
    else:
        device = -1  # Default to CPU if MPS is not available
        print("MPS not available. Using CPU.")
    

    text_column = 'review_body'
    

    models = {
        "LiYuan/amazon-review-sentiment-analysis": pipeline(
            "sentiment-analysis",
            model="LiYuan/amazon-review-sentiment-analysis",
            use_fast=True,
            device=device  
        )
    }
    

    model_key = "LiYuan/amazon-review-sentiment-analysis"
    df_english = analyze_sentiments(df_english, text_column=text_column, model_key=model_key, models=models)
    df_french = analyze_sentiments(df_french, text_column=text_column, model_key=model_key, models=models)
    df_spanish = analyze_sentiments(df_spanish, text_column=text_column, model_key=model_key, models=models)




#################### evaluating liyuan sentiment 


import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


labels = ['negative', 'neutral', 'positive']


def map_liyuan_to_sentiment(label):
    
    star = int(label.split()[0])  
    

    if star in [1, 2]:
        return 'negative'
    elif star == 3:
        return 'neutral'
    elif star in [4, 5]:
        return 'positive'
    else:
        print(f"Unexpected label: {label}") 
        return None  


for df in [df_english, df_french, df_spanish]:
    df['mapped_sentiment'] = df['LiYuan/amazon-review-sentiment-analysis_label'].apply(map_liyuan_to_sentiment)
    df['sentiment'] = df['sentiment'].str.lower()  


def plot_confusion_matrix(df, true_column, pred_column, labels, title):
    cm = confusion_matrix(df[true_column], df[pred_column], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


for language, df in [('English', df_english), ('French', df_french), ('Spanish', df_spanish)]:
    print(f"\nConfusion Matrix and Metrics for {language} Reviews:")
    print("Unique values in true labels (sentiment):", df['sentiment'].unique())
    print("Unique values in predicted labels (mapped_sentiment):", df['mapped_sentiment'].unique())
    

    cm = confusion_matrix(df['sentiment'], df['mapped_sentiment'], labels=labels)
    print(f"\nConfusion Matrix for {language} Reviews:")
    print(cm)
    

    plot_confusion_matrix(
        df,
        true_column='sentiment',
        pred_column='mapped_sentiment',
        labels=labels,
        title=f"Confusion Matrix for {language} Reviews"
    )
    

    print(f"\nClassification Report for {language} Reviews:")
    print(classification_report(
        df['sentiment'], 
        df['mapped_sentiment'], 
        target_names=labels, 
        zero_division=0
    ))
    

    accuracy_sklearn = accuracy_score(df['sentiment'], df['mapped_sentiment'])
    print(f"Accuracy (from sklearn) for {language} Reviews: {accuracy_sklearn:.2f}")
    
    
    

    
    

##################### sentiment analysis unsing "cardiffnlp/twitter-xlm-roberta-base-sentiment"
def analyze_sentiments(df, text_column, model_key, models):


    sentiment_model = models[model_key]
    

    sentiment_results = df[text_column].apply(lambda x: sentiment_model(x)[0])  # Apply model on each row
    

    df[f'{model_key}_label'] = sentiment_results.apply(lambda x: x['label'])
    df[f'{model_key}_score'] = sentiment_results.apply(lambda x: x['score'])
    
    return df


if __name__ == "__main__":
    import torch
    from transformers import pipeline


    if torch.backends.mps.is_available():
        device = 0
        print("Using MPS for hardware acceleration.")
    else:
        device = -1  
        print("MPS not available. Using CPU.")
    

    text_column = 'review_body'
    

    models = {
        "cardiffnlp/twitter-xlm-roberta-base-sentiment": pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            use_fast=True,
            device=device  
        )
    }
    

    model_key = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    df_english = analyze_sentiments(df_english, text_column=text_column, model_key=model_key, models=models)
    df_french = analyze_sentiments(df_french, text_column=text_column, model_key=model_key, models=models)
    df_spanish = analyze_sentiments(df_spanish, text_column=text_column, model_key=model_key, models=models)


#############  evaluating XLM roberta  sentiment 

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


labels = ['negative', 'neutral', 'positive']


def plot_confusion_matrix(df, true_column, pred_column, labels, title):
    cm = confusion_matrix(df[true_column], df[pred_column], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


for language, df in [('English', df_english), ('French', df_french), ('Spanish', df_spanish)]:
    print(f"\nConfusion Matrix and Metrics for {language} Reviews:")
    print("Unique values in true labels (sentiment):", df['sentiment'].unique())
    print("Unique values in predicted labels (XLM-RoBERTa_sentiment_label):", df['cardiffnlp/twitter-xlm-roberta-base-sentiment_label'].unique())
    

    cm = confusion_matrix(df['sentiment'], df['cardiffnlp/twitter-xlm-roberta-base-sentiment_label'], labels=labels)
    print(f"\nConfusion Matrix for {language} Reviews:")
    print(cm)
    

    plot_confusion_matrix(
        df,
        true_column='sentiment',
        pred_column='cardiffnlp/twitter-xlm-roberta-base-sentiment_label',
        labels=labels,
        title=f"Confusion Matrix for {language} Reviews"
    )
    

    print(f"\nClassification Report for {language} Reviews:")
    print(classification_report(
        df['sentiment'], 
        df['cardiffnlp/twitter-xlm-roberta-base-sentiment_label'], 
        target_names=labels, 
        zero_division=0
    ))
    

    accuracy_sklearn = accuracy_score(df['sentiment'], df['cardiffnlp/twitter-xlm-roberta-base-sentiment_label'])
    print(f"Accuracy (from sklearn) for {language} Reviews: {accuracy_sklearn:.2f}")
    

############ TOPIC MODELLING 




#############lemmatizing the review for topic modelling 
#first removing stopwords with nltk and spacy     
####################### cleaned no stoword english 

import spacy
from nltk.corpus import stopwords
import pandas as pd


nlp_english = spacy.load("en_core_web_sm")


nltk_stopwords_english = set(stopwords.words("english"))


spacy_stopwords_english = nlp_english.Defaults.stop_words


combined_stopwords_english = nltk_stopwords_english.union(spacy_stopwords_english)
def remove_stopwords_english(review):
    if isinstance(review, list):  
        return [word for word in review if word.lower() not in combined_stopwords_english]
    elif isinstance(review, str): 
        doc = nlp_english(review)
        return [token.text for token in doc if token.text.lower() not in combined_stopwords_english and token.is_alpha]


df_english['cleaned_no_stopwords'] = df_english['cleaned_reviews'].apply(remove_stopwords_english)

  
    
################### cleaned no stoword spanish 

import spacy
from nltk.corpus import stopwords
import pandas as pd


nlp = spacy.load("es_core_news_sm")
import nltk
nltk.download('stopwords')


nltk_stopwords = set(stopwords.words("spanish"))


spacy_stopwords = nlp.Defaults.stop_words
combined_stopwords = nltk_stopwords.union(spacy_stopwords)


def remove_stopwords(review):
    if isinstance(review, list):
        return [word for word in review if word.lower() not in combined_stopwords]
    elif isinstance(review, str): 
        doc = nlp(review)
        return [token.text for token in doc if token.text.lower() not in combined_stopwords and token.is_alpha]


df_spanish['cleaned_no_stopwords'] = df_spanish['cleaned_reviews'].apply(remove_stopwords)
    
########################$ cleaned no stopword french 


import spacy
from nltk.corpus import stopwords
import pandas as pd


nlp = spacy.load("fr_core_news_sm")

import nltk
nltk.download('stopwords')


nltk_stopwords = set(stopwords.words("french"))


spacy_stopwords = nlp.Defaults.stop_words
combined_stopwords = nltk_stopwords.union(spacy_stopwords)


def remove_stopwords(review):
    if isinstance(review, list):
        return [word for word in review if word.lower() not in combined_stopwords]
    elif isinstance(review, str): 
        doc = nlp(review)
        return [token.text for token in doc if token.text.lower() not in combined_stopwords and token.is_alpha]
    



df_french['cleaned_no_stopwords'] = df_french['cleaned_reviews'].apply(remove_stopwords)




############## Lemmatization function for French spanish, and english 
# Load SpaCy's Spanish model
nlp_spanish = spacy.load("es_core_news_sm")
nlp_french = spacy.load("fr_core_news_sm")
nlp_english = spacy.load("en_core_web_sm")


def lemmatize_review_french(review):
    if isinstance(review, list):
        doc = nlp_french(" ".join(review)) 
    elif isinstance(review, str):  
        doc = nlp_french(review)
    else:
        return []  
    
    return [token.lemma_ for token in doc if token.is_alpha]


def lemmatize_review_spanish(review):
    if isinstance(review, list): 
        doc = nlp_spanish(" ".join(review))  
    elif isinstance(review, str):  
        doc = nlp_spanish(review)
    else:
        return []  
    
    return [token.lemma_ for token in doc if token.is_alpha]


def lemmatize_review_english(review):
    if isinstance(review, list): 
        doc = nlp_english(" ".join(review))  
    elif isinstance(review, str): 
        doc = nlp_english(review)
    else:
        return []  
    
    return [token.lemma_ for token in doc if token.is_alpha]


# Lemmatize French cleaned_no_stopwords
df_french['lemmatized_reviews'] = df_french['cleaned_no_stopwords'].apply(lemmatize_review_french)

# Lemmatize Spanish cleaned_no_stopwords
df_spanish['lemmatized_reviews'] = df_spanish['cleaned_no_stopwords'].apply(lemmatize_review_spanish)

df_english['lemmatized_reviews'] = df_english['cleaned_no_stopwords'].apply(lemmatize_review_english)


neutral_reviews_count = df_english[df_english['sentiment'] == 'negative'].shape[0]

print(f"Number of neutral reviews: {neutral_reviews_count}")

#reimporting the dataset with sentiment clsasifier already runed

import pandas as pd
df_spanish= pd.read_csv(r"/Users/inesbenhamza/df_spanish2.csv")
df_english= pd.read_csv(r"/Users/inesbenhamza/df_english2.csv")
df_french= pd.read_csv(r"/Users/inesbenhamza/df_french2.csv")



#######################  droppping reviews where score is too different than ratings (2 stars difference)


def map_stars_to_numeric(label):
    return int(label.split()[0]


df_english['bert_numeric_rating'] = df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'].apply(map_stars_to_numeric)
df_english['rating_difference'] = abs(df_english['bert_numeric_rating'] - df_english['stars'])
threshold = 2
df_english = df_english[df_english['rating_difference'] <= threshold].copy()
df_english.drop(columns=['bert_numeric_rating', 'rating_difference'], inplace=True)


df_spanish['bert_numeric_rating'] = df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'].apply(map_stars_to_numeric)
df_spanish['rating_difference'] = abs(df_spanish['bert_numeric_rating'] - df_spanish['stars'])
df_spanish = df_spanish[df_spanish['rating_difference'] <= threshold].copy()
df_spanish.drop(columns=['bert_numeric_rating', 'rating_difference'], inplace=True)


df_french['bert_numeric_rating'] = df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'].apply(map_stars_to_numeric)
df_french['rating_difference'] = abs(df_french['bert_numeric_rating'] - df_french['stars'])
df_french = df_french[df_french['rating_difference'] <= threshold].copy()
df_french.drop(columns=['bert_numeric_rating', 'rating_difference'], inplace=True)

# Print results
print(f"The filtered English reviews DataFrame has {filtered_english_reviews.shape[0]} rows.")
print(f"The filtered Spanish reviews DataFrame has {filtered_spanish_reviews.shape[0]} rows.")
print(f"The filtered French reviews DataFrame has {filtered_french_reviews.shape[0]} rows.")

print(f"The English reviews DataFrame has {df_english.shape[0]} rows.")
print(f"The Spanish reviews DataFrame has {df_spanish.shape[0]} rows.")
print(f"The French reviews DataFrame has {df_french.shape[0]} rows.")

#The filtered_reviews DataFrame has 24587 columns.
#The english DataFrame has 24994 columns.


#The filtered English reviews DataFrame has 24994 rows.
#The filtered Spanish reviews DataFrame has 24997 rows.
#The filtered French reviews DataFrame has 24998 rows.


#The  English reviews DataFrame has 24587 rows.
#The  Spanish reviews DataFrame has 24636 rows.
#The French reviews DataFrame has 24692 rows.






###############english LDA

######################## Topic modelling on one star and 2 stars reviews using nlptown bert and LDA


low_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]



vectorizer = CountVectorizer(max_df=0.80, min_df=30, stop_words='english')
dtm = vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])


# Apply LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)  
lda.fit(dtm)

# Get the topics
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


feature_names = vectorizer.get_feature_names_out()

# Display the topics
topics = display_topics(lda, feature_names, 10)  

# Print the topics
for topic, words in topics.items():
    print(f"{topic}: {', '.join(words)}")


#########################

####coherence score for optimal number of topics : gridsearch

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt


low_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]


texts = [review.split() for review in low_star_reviews['lemmatized_reviews']]


id2word = Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]


topic_numbers = range(2, 21)
coherence_scores = []


for num_topics in topic_numbers:
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42)
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_scores.append(coherence_model.get_coherence())


plt.figure(figsize=(10, 6))
plt.plot(topic_numbers, coherence_scores, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.title('Optimal Number of Topics Based on Coherence Score')
plt.grid()
plt.show()


optimal_topics = topic_numbers[coherence_scores.index(max(coherence_scores))]
print(f"Optimal number of topics: {optimal_topics}")

#got best at 5 topics


###########coherence scre for 5 topics 

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt


low_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]


texts = [review.split() for review in low_star_reviews['lemmatized_reviews']]


id2word = Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]


num_topics = 3  
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42)
coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_score = coherence_model.get_coherence()

print(f"Coherence Score for {num_topics} topics: {coherence_score}")


###visualization 


import matplotlib.pyplot as plt
import numpy as np

def visualize_topics_grid(model, feature_names, no_top_words=10, no_topics=3):

    # Set up the grid layout
    fig, axes = plt.subplots(1, no_topics, figsize=(20, 6), sharey=False)
    fig.suptitle("Topic Word Scores", fontsize=16)
    
    
    for topic_idx, topic in enumerate(model.components_[:no_topics]):
        top_features_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features_idx]
        top_weights = topic[top_features_idx]
        
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, color=plt.cm.tab10(topic_idx / no_topics))
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=14)
        ax.invert_yaxis()  
        ax.set_xlabel("Weight")
        ax.set_xticks(np.arange(0, max(top_weights) + 200, step=200))
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()

# Example usage for LDA
lda_feature_names = vectorizer.get_feature_names_out()
visualize_topics_grid(lda, lda_feature_names, no_top_words=10, no_topics=3)



#### ####################french LDA
###########
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

low_star_reviews = df_french[
    (df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]



custom_stop_words = ['jai', 'cest']


vectorizer = CountVectorizer(max_df=0.80, min_df=20, stop_words=custom_stop_words)
dtm = vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])


lda = LatentDirichletAllocation(n_components=3, random_state=42)  
lda.fit(dtm)

def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics

feature_names = vectorizer.get_feature_names_out()


topics = display_topics(lda, feature_names, 10)  

for topic, words in topics.items():
    print(f"{topic}: {', '.join(words)}")
    



############ visualization of the topics 

import matplotlib.pyplot as plt
import numpy as np

def visualize_topics_grid(model, feature_names, no_top_words=10, no_topics=3):


    fig, axes = plt.subplots(1, no_topics, figsize=(20, 6), sharey=False)
    fig.suptitle("Topic Word Scores", fontsize=16)
    

    for topic_idx, topic in enumerate(model.components_[:no_topics]):
        top_features_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features_idx]
        top_weights = topic[top_features_idx]
        
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, color=plt.cm.tab10(topic_idx / no_topics))
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=14)
        ax.invert_yaxis()  
        ax.set_xlabel("Weight")
        ax.set_xticks(np.arange(0, max(top_weights) + 200, step=200))
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()

# Example usage for LDA
lda_feature_names = vectorizer.get_feature_names_out()
visualize_topics_grid(lda, lda_feature_names, no_top_words=10, no_topics=3)

 



####################### spanish LDA


low_star_reviews = df_spanish[
    (df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]

custom_stop_words = ['él']


vectorizer = CountVectorizer(max_df=0.80, min_df=20, stop_words=custom_stop_words)
dtm = vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])


lda = LatentDirichletAllocation(n_components=5, random_state=989)  
lda.fit(dtm)


def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


feature_names = vectorizer.get_feature_names_out()

topics = display_topics(lda, feature_names, 10)  


for topic, words in topics.items():
    print(f"{topic}: {', '.join(words)}")
    
    

import matplotlib.pyplot as plt
import numpy as np

def visualize_topics_grid(model, feature_names, no_top_words=10, no_topics=3):

    # Set up the grid layout
    fig, axes = plt.subplots(1, no_topics, figsize=(20, 6), sharey=False)
    fig.suptitle("Topic Word Scores", fontsize=16)
    

    for topic_idx, topic in enumerate(model.components_[:no_topics]):
        top_features_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features_idx]
        top_weights = topic[top_features_idx]
        
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, color=plt.cm.tab10(topic_idx / no_topics))
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=14)
        ax.invert_yaxis()  
        ax.set_xlabel("Weight")
        ax.set_xticks(np.arange(0, max(top_weights) + 200, step=200))
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()

# Example usage for LDA
lda_feature_names = vectorizer.get_feature_names_out()
visualize_topics_grid(lda, lda_feature_names, no_top_words=10, no_topics=3)


###########       NMF 

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

####################### english NMF
low_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]



tfidf_vectorizer = TfidfVectorizer(max_df=0.70, min_df=20, stop_words='english')
tfidf_dtm = tfidf_vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])

# Apply NMF
nmf = NMF(n_components=3, random_state=76)  
nmf.fit(tfidf_dtm)


def display_nmf_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


nmf_topics = display_nmf_topics(nmf, tfidf_feature_names, 10)  # Displaying top 10 words for each topic

# Print the topics
print("\nNMF Topics:")
for topic, words in nmf_topics.items():
    print(f"{topic}: {', '.join(words)}")
    
    
########### frnech  NMF

low_star_reviews = df_french[
    (df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]

custom_stop_words = ['jai']


tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=30, stop_words = custom_stop_words )
tfidf_dtm = tfidf_vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])

nmf = NMF(n_components=3, random_state=42)  
nmf.fit(tfidf_dtm)


def display_nmf_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


nmf_topics = display_nmf_topics(nmf, tfidf_feature_names, 10)  


print("\nNMF Topics:")
for topic, words in nmf_topics.items():
    print(f"{topic}: {', '.join(words)}")
    
    
########### spanish  NMF



low_star_reviews = df_spanish[
    (df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]

custom_stop_words = ['él']

tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=20,stop_words = custom_stop_words )
tfidf_dtm = tfidf_vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])


nmf = NMF(n_components=3, random_state=42)  
nmf.fit(tfidf_dtm)


def display_nmf_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


nmf_topics = display_nmf_topics(nmf, tfidf_feature_names, 10)  

print("\nNMF Topics:")
for topic, words in nmf_topics.items():
    print(f"{topic}: {', '.join(words)}")



#LSA english 


low_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]

tfidf_vectorizer = TfidfVectorizer(max_df=0.70, min_df=20, stop_words='english')
tfidf_dtm = tfidf_vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])

from sklearn.decomposition import TruncatedSVD


lsa = TruncatedSVD(n_components=5, random_state=42)
lsa.fit(tfidf_dtm)


def display_lsa_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


lsa_topics = display_lsa_topics(lsa, tfidf_feature_names, 10) 


print("\nLSA Topics:")
for topic, words in lsa_topics.items():
    print(f"{topic}: {', '.join(words)}")
    
    
############visualization 

import matplotlib.pyplot as plt
import numpy as np

def visualize_topics_grid_lsa(model, feature_names, no_top_words=10, no_topics=3):


    fig, axes = plt.subplots(1, no_topics, figsize=(20, 6), sharey=False)
    fig.suptitle("Topic Word Scores (LSA)", fontsize=16)
    

    for topic_idx, topic in enumerate(model.components_[:no_topics]):
        top_features_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features_idx]
        top_weights = topic[top_features_idx]
        
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, color=plt.cm.tab10(topic_idx / no_topics))
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=14)
        ax.invert_yaxis() 
        ax.set_xlabel("Weight")
        ax.set_xlim(0, max(top_weights) * 1.1)  
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()

lsa_feature_names = tfidf_vectorizer.get_feature_names_out()
visualize_topics_grid_lsa(lsa, lsa_feature_names, no_top_words=10, no_topics=3)





    
##########   LSA spanish 
    
low_star_reviews = df_spanish[
    (df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]



tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=20)
tfidf_dtm = tfidf_vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])

from sklearn.decomposition import TruncatedSVD


lsa = TruncatedSVD(n_components=3, random_state=42)
lsa.fit(tfidf_dtm)

def display_lsa_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

lsa_topics = display_lsa_topics(lsa, tfidf_feature_names, 10)  

print("\nLSA Topics:")
for topic, words in lsa_topics.items():
    print(f"{topic}: {', '.join(words)}")
    

##visualizaiton 
import matplotlib.pyplot as plt
import numpy as np

def visualize_topics_grid_lsa(model, feature_names, no_top_words=10, no_topics=3):


    fig, axes = plt.subplots(1, no_topics, figsize=(20, 6), sharey=False)
    fig.suptitle("Topic Word Scores (LSA)", fontsize=16)
    

    for topic_idx, topic in enumerate(model.components_[:no_topics]):
        top_features_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features_idx]
        top_weights = topic[top_features_idx]
        
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, color=plt.cm.tab10(topic_idx / no_topics))
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=14)
        ax.invert_yaxis() 
        ax.set_xlabel("Weight")
        ax.set_xlim(0, max(top_weights) * 1.1)  
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()


lsa_feature_names = tfidf_vectorizer.get_feature_names_out()
visualize_topics_grid_lsa(lsa, lsa_feature_names, no_top_words=10, no_topics=3)






########################## LSA french 


low_star_reviews = df_french[
    (df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') | 
    (df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '2 stars')
]

tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=20)
tfidf_dtm = tfidf_vectorizer.fit_transform([str(text) for text in low_star_reviews['lemmatized_reviews']])

from sklearn.decomposition import TruncatedSVD


lsa = TruncatedSVD(n_components=3, random_state=42)
lsa.fit(tfidf_dtm)


def display_lsa_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


lsa_topics = display_lsa_topics(lsa, tfidf_feature_names, 10)  

print("\nLSA Topics:")
for topic, words in lsa_topics.items():
    print(f"{topic}: {', '.join(words)}")
    
    

    
###################### WORD CLOUD applied on tokenized review without stopwords 





########## 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# for 1-star and 2-star reviews in the apparel category
low_star_apparel_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'].isin(['1 star', '2 stars'])) & 
    (df_english['product_category'] == 'apparel')
]


stopwords = set(STOPWORDS)  #
text = " ".join(" ".join(token for token in review if token.lower() not in stopwords and token != "'") 
                for review in low_star_apparel_reviews['cleaned_no_stopwords'])


wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=200,
    colormap='coolwarm'
).generate(text)


plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.title("Word Cloud for 1- and 2-Star Apparel Reviews", fontsize=18)
plt.show()


######### creating cleaned_no_stopwordswhich is tokenized and remove stopwords from both spacy and nltk 
#created before
############ wordcloud french applied on cleaned reviews, no stopwords 

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Filter for 1-star and 2-star reviews in the apparel category
low_star_apparel_reviews = df_french[
    (df_french['nlptown/bert-base-multilingual-uncased-sentiment_label'].isin(['1 star', '2 stars'])) & 
    (df_french['product_category'] == 'apparel')
]


stopwords = set(STOPWORDS)  
stopwords.update(["apparel", "product"])  
text = " ".join(
    " ".join(
        token for token in review if token.lower() not in stopwords and token.isalpha()
    )
    for review in low_star_apparel_reviews['cleaned_no_stopwords']
)

#
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=200,
    stopwords=stopwords, 
    colormap='coolwarm'
).generate(text)


plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.title("Word Cloud for 1- and 2-Star Apparel Reviews", fontsize=18)
plt.show()



###########spanish 


############ WordCloud for Spanish Reviews, applied on cleaned reviews, no stopwords 

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


low_star_apparel_reviews = df_spanish[
    (df_spanish['nlptown/bert-base-multilingual-uncased-sentiment_label'].isin(['1 star', '2 stars'])) & 
    (df_spanish['product_category'] == 'apparel')
]


stopwords = set(STOPWORDS)  
stopwords.update(["apparel", "producto"])  
text = " ".join(
    " ".join(
        token for token in review if token.lower() not in stopwords and token.isalpha()
    )
    for review in low_star_apparel_reviews['cleaned_no_stopwords']
)


wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=200,
    stopwords=stopwords,   
    colormap='coolwarm'
).generate(text)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.title("Word Cloud for 1- and 2-Star Apparel Reviews (Spanish)", fontsize=18)
plt.show()






###for shoes with lda

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


apparel_one_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'].isin(['1 star'])) & 
    (df_english['product_category'] == 'shoes')  # Replace 'shoes' with the category u would like to analyze 
]


count_vectorizer = CountVectorizer(max_df=0.6, stop_words='english')
count_dtm = count_vectorizer.fit_transform([str(text) for text in apparel_one_star_reviews['lemmatized_reviews']])


lda = LatentDirichletAllocation(n_components=3, random_state=42)  
lda.fit(count_dtm)


def display_lda_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


count_feature_names = count_vectorizer.get_feature_names_out()

lda_topics = display_lda_topics(lda, count_feature_names, 10)  


print("\nLDA Topics for 1-Star and 2-Star Reviews in Apparel:")
for topic, words in lda_topics.items():
    print(f"{topic}: {', '.join(words)}")


################shoes with nmf


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


shoes_one_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'].isin(['1 star', '2 stars'])) & 
    (df_english['product_category'] == 'shoes')  # Specify 'shoes' category
]


tfidf_vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
tfidf_dtm = tfidf_vectorizer.fit_transform([str(text) for text in shoes_one_star_reviews['lemmatized_reviews']])


nmf = NMF(n_components=3, random_state=9, init='nndsvd')  
nmf.fit(tfidf_dtm)


def display_nmf_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


nmf_topics = display_nmf_topics(nmf, tfidf_feature_names, 10)  


print("\nNMF Topics for 1-Star and 2-Star Reviews in Shoes:")
for topic, words in nmf_topics.items():
    print(f"{topic}: {', '.join(words)}")


def visualize_topics_grid(model, feature_names, no_top_words=10, no_topics=3):

    fig, axes = plt.subplots(1, no_topics, figsize=(20, 6), sharey=False)
    fig.suptitle("NMF Topic Word Scores", fontsize=16)
    

    for topic_idx, topic in enumerate(model.components_[:no_topics]):
        top_features_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features_idx]
        top_weights = topic[top_features_idx]
        
        ax = axes[topic_idx]
        ax.barh(top_words, top_weights, color=plt.cm.tab10(topic_idx / no_topics))
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=14)
        ax.invert_yaxis()  
        ax.set_xlabel("Weight")
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()


visualize_topics_grid(nmf, tfidf_feature_names, no_top_words=10, no_topics=3)   
    
    
    
    
    
# Filter dataset for negative reviews in the "electronics" category 


electronics_five_star_reviews = df_english[
    (df_english['nlptown/bert-base-multilingual-uncased-sentiment_label'] == '1 star') & 
    (df_english['product_category'] == 'apparel') 
]




vectorizer = CountVectorizer(max_df=0.8, min_df=10, stop_words='english')
dtm = vectorizer.fit_transform([str(text) for text in electronics_five_star_reviews ['lemmatized_reviews']])


lda = LatentDirichletAllocation(n_components=10, random_state=123)  
lda.fit(dtm)


def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics

feature_names = vectorizer.get_feature_names_out()


topics = display_topics(lda, feature_names, 5)  


print("\nLDA Topics for 5-Star Reviews in apparel:")
for topic, words in topics.items():
    print(f"{topic}: {', '.join(words)}")
    
    
  
    
    


# Save DataFrames as CSV files
df_english.to_csv('df_english2.csv', index=False)
df_french.to_csv('df_french2.csv', index=False)
df_spanish.to_csv('df_spanish2.csv', index=False)

print("DataFrames saved as CSV files:")
print("1. df_english.csv")
print("2. df_french.csv")
print("3. df_spanish.csv")

