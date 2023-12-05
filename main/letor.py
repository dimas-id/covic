import lightgbm as lgb
import numpy as np
import random
import os
import pickle

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

class Letor:
    def __init__(self, train_docs_path, train_queries_path, train_qrels_path, num_latent_topics=200):
        self.documents = self.read_documents(train_docs_path)
        self.queries = self.read_queries(train_queries_path)
        self.q_docs_rel = self.process_qrels(train_qrels_path)
        self.num_latent_topics = num_latent_topics
        self.dictionary, self.bow_corpus, self.model = self.build_lsi_model()
        self.ranker = None

    def read_documents(self, file_path):
        documents = {}
        with open(file_path, encoding='utf-8') as file:
            for line in file:
                split_line = line.split()
                doc_id, content = split_line[0], split_line[1:]
                documents[doc_id] = content
        return documents
    
    def read_queries(self, file_path):
            queries = {}
            with open(file_path, encoding='utf-8') as file:
                for line in file:
                    split_line = line.split()
                    q_id, content = split_line[0], split_line[1:]
                    queries[q_id] = content
            return queries
    
    def process_qrels(self, file_path):
            # Membuat dataset untuk training dengan format:

            # [(query_text, document_text, relevance), ...]
            #
            # relevance awalnya bernilai 1, 2, 3 --> tidak perlu dinormalisasi
            # biarkan saja integer (syarat dari library LightGBM untuk
            # LambdaRank)
            #
            # relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant)
            
            q_docs_rel = {} # grouping by q_id
            # Format --> Qid: (doc_id, relevance)
            with open(file_path, encoding='utf-8') as file:
                for line in file:
                    q_id, doc_id, rel = line.split()
                    if (q_id in self.queries) and (doc_id in self.documents):
                        if q_id not in q_docs_rel:
                            q_docs_rel[q_id] = []
                        q_docs_rel[q_id].append((doc_id, int(rel)))
            return q_docs_rel
    
    def build_lsi_model(self):
        dictionary = Dictionary()
        bow_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]
        model = LsiModel(bow_corpus, num_topics=self.num_latent_topics)
        return dictionary, bow_corpus, model

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

    def build_dataset(self, num_negatives=1):
        group_qid_count = []
        dataset = []
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            group_qid_count.append(len(docs_rels) + num_negatives)
            for doc_id, rel in docs_rels:
                dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

        X = []
        Y = []
        for (query, doc, rel) in dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y, group_qid_count

    def train_ranker(self, X, Y, group_qid_count):
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100,
            importance_type="gain",
            metric="ndcg",
            num_leaves=40,
            learning_rate=0.02,
            max_depth=-1
        )
        print("Labels (Y) before training:", Y)
        ranker.fit(X, Y, group=group_qid_count)
        return ranker

    def predict_rankings(self, query, docs):
        X_unseen = [self.features(query.split(), doc.split()) for (_, doc) in docs]
        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)
        did_scores = list(zip([did for (did, _) in docs], scores))
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        return sorted_did_scores
    
    def train_letor(self, save_model_path=None):
        X, Y, group_qid_count = self.build_dataset()
        print('Train model start!!!!!!')
        self.ranker = self.train_ranker(X, Y, group_qid_count)

        if save_model_path:
            self.save_model(save_model_path)
            print('model saved successfully!!!!!')
        
        print('Train model finish')

        # return ranker
    
    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.ranker, file)

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.ranker = pickle.load(file)
            print('model loaded')
            print(self.ranker)
            return self.ranker

if __name__ == "__main__":
    # Contoh pemanggilan letor (sesuai dengan tutorial notebook letor)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    # train_docs_path = os.path.join(current_directory,"static/data/tes_merge_0.txt")
    train_docs_path = os.path.join(current_directory,"static/data/docs.txt")
    print(train_docs_path)
    train_queries_path = os.path.join(current_directory,"static/data/queries.txt")
    train_qrels_path = os.path.join(current_directory,"static/data/qrels.txt")

    model_path = os.path.join(current_directory, 'static','model', 'ranker_model.pkl')

    letor = Letor(train_docs_path, train_queries_path, train_qrels_path)
    ranker = letor.train_letor(save_model_path=model_path)

    letor = Letor(train_docs_path, train_queries_path, train_qrels_path)
    letor.load_model(model_path=model_path)
    # with open(model_path, 'rb') as file:
    #         ranker = pickle.load(file)

    query_to_predict = "how much cancer risk can be avoided through lifestyle change ?"
    docs_to_predict = [("D1", "dietary restriction reduces insulin-like growth factor levels..."),
                    ("D2", "study hard as your blood boils"),
                    ("D3", "processed meats risk childhood leukemia california usa..."),
                    ("D4", "long-term effects calorie protein restriction serum igf num..."),
                    ("D5", "cancer preventable disease requires major lifestyle...")]

    rankings = letor.predict_rankings(query_to_predict, docs_to_predict)
    print(f'type rankings: {type(rankings)}')

    print("Query:", query_to_predict)
    print("Rankings:")
    for (doc_id, score) in rankings:
        print(doc_id, score)