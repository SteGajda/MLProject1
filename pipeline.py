from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import numpy as np

class CleanCommentTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Comment'] = df['Comment'].fillna('')

        def preprocess_text(text):
            text = text.lower()
            text = ''.join([char for char in text if char not in string.punctuation])
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            return " ".join(tokens)

        df['Clean_Comment'] = df['Comment'].apply(preprocess_text)
        return df['Clean_Comment'].values


class BoWwithNgramTransform(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(1, 2)):
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)

    def fit(self, X, y=None):
        self.vectorizer.fit(X.ravel())
        return self

    def transform(self, X):
        return self.vectorizer.transform(X.ravel()).toarray()


class TextEmbedderTransform(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()


class CombinedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bow = BoWwithNgramTransform()
        self.embed = TextEmbedderTransform()

    def fit(self, X, y=None):
        self.bow.fit(X)
        self.embed.fit(X)
        return self

    def transform(self, X):
        bow_feats = self.bow.transform(X)
        embed_feats = self.embed.transform(X)
        return np.hstack([bow_feats, embed_feats])


full_pipeline = Pipeline([
    ('clean_text', CleanCommentTransform()),
    ('features', CombinedFeatures())
])
