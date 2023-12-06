import lightgbm as lgb
import numpy as np
import random
import os
import pickle

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

class LetorTest:
    def __init__(self, dict_path, model_path, lsi_path, num_latent_topics=200):
        
        self.dictionary = Dictionary.load(dict_path)
        self.model = LsiModel.load(lsi_path)
        self.ranker = lgb.Booster(model_file=model_path)
        self.num_latent_topics = num_latent_topics

    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.num_latent_topics else [0.] * self.num_latent_topics

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def predict_rankings(self, query, docs):
        X_unseen = [self.features(query.split(), doc.split()) for (_, doc) in docs]
        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)
        did_scores = list(zip([did for (did, _) in docs], scores))
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        return sorted_did_scores
