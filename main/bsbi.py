import os
import pickle
import contextlib
import heapq
import math

from .index import InvertedIndexReader, InvertedIndexWriter
from .util import IdMap, merge_and_sort_posts_and_tfs
from .compression import VBEPostings
from tqdm import tqdm

import re

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords')


from operator import itemgetter


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.avg_doc_length = None

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        td_pairs = []
        tokenizer_pattern = r'\w+'

        stemmer = PorterStemmer()
        stopwords_set = set(stopwords.words("english"))

        for root, _, files in os.walk(os.path.join(self.data_dir,block_path)):
            for file in files:
                with open(os.path.join(root,file),'r', encoding='utf-8') as f:
                    text = f.read()
                    text = re.findall(tokenizer_pattern,text.lower())
                    text = [stemmer.stem(word) for word in text]
                    text = [word for word in text if word not in stopwords_set]

                    for token in text:
                        term_id = self.term_id_map[token]
                        doc_id = self.doc_id_map[os.path.join(self.data_dir, block_path, file)]
                        td_pairs.append((term_id,doc_id))
        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {} # key: term-id, value: sorted doc-id: freq. term-id muncul pada doc-id
        # {term-id: {doc-id: freq}} -> term_dict, {doc-id: freq} -> docid_freq
        
        for term_id, doc_id in td_pairs:
            # create or get docid_freq
            # asumsi: ketika term_id ditambahkan ke term_dict, maka docid_freq sudah juga ditambahkan
            if term_id not in term_dict:
                docid_freq = {}
            else:
                docid_freq = term_dict[term_id] # case term_id sudah ada
            # increment/count jumlah kemunculan term_id pada doc_id
            if doc_id not in docid_freq:
                docid_freq[doc_id] = 1
            else:
                docid_freq[doc_id] = docid_freq[doc_id]+1
            term_dict[term_id] = docid_freq
        
        # format: {1: {101:2,104:4}, 2: {101:1,105:5}}
        for term_id in sorted(term_dict.keys()): # iterate sorted term-id
            # sort doc_id
            docid_freq = {docid: term_dict[term_id][docid]
                          for docid in sorted(term_dict[term_id].keys())}

            index.append(term_id, list(docid_freq.keys()), list(docid_freq.values()))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load() # load terms and docs from pickle file

        tokenizer_pattern = r'\w+'
        stemmer = PorterStemmer()
        stopwords_set = set(stopwords.words("english"))

        query = re.findall(tokenizer_pattern,query.lower())
        query = [stemmer.stem(word) for word in query]
        query = [word for word in query if word not in stopwords_set]

        if len(query) == 0: # case when all of the query are in stopwords, hence the length of list of token in query is 0
            return []
        
        if len(query) == 1 and query[0] not in self.term_id_map:
            return []

        selected_token = []
        for token in query:
            if token not in self.term_id_map:
                continue
            selected_token.append(self.term_id_map[token])
        query = selected_token

        # print(f'Query: {query}')
        
        scores = {} # key: docid, val: score
        res = []
        with InvertedIndexReader(index_name=self.index_name, postings_encoding=self.postings_encoding, directory=self.output_dir) as reader:
            for token in query:
                # print(f'token: {self.term_id_map[token]}')
                
                postings_lst = reader.get_postings_list(token) # doc_id, tf(t,D)
                # print(postings_lst)

                N = len(reader.doc_length)
                # print(N)

                dft = len(postings_lst[0]) # total document in collection that contain the term t
                # print(f'dft: {dft}')
                wtq = math.log10(N/dft) # w(t, Q) a.k.a idf
                # print(f'wtq: {wtq}')

                for (docid, tf) in zip(*postings_lst):
                    # print(f'(docid, tf): {(docid, tf)}')
                    wtd = (1+math.log10(tf)) if tf > 0 else 0 # W(t,D)
                    # print(f'wtd: {wtd}')

                    # tf.idf -> W(t,D).W(t,Q) => (1+log(tf)).log(N/df)

                    if docid not in scores:
                        scores[docid] = 0
                    scores[docid] += wtd*wtq # tf*idf, akumulasi seluruh tf*idf dari tiap docid dan query


            top_k = list(heapq.nlargest(k, scores.items(), key=lambda x: x[1])) # get top k highest score document, compare using x[1]: score of each doc
            # print(f'top_k: {top_k}')

            for docid, score in top_k:
                res.append((score, self.doc_id_map[docid]))
        return res

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO
        # formula:
        # log(N/dft).(((k1+1).tft)/(k1.((1-b)+b.(dl/avgdl))+tft))
        # idf.rsv-norm -> rsv = ((k1+1).tft)/(k1+tft), doc length norm: (1-b)+b.(dl/avgdl)

        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load() # load terms and docs from pickle file

        tokenizer_pattern = r'\w+'
        stemmer = PorterStemmer()
        stopwords_set = set(stopwords.words("english"))

        query = re.findall(tokenizer_pattern,query.lower())
        query = [stemmer.stem(word) for word in query]
        query = [word for word in query if word not in stopwords_set]

        if len(query) == 0: # case when all of the query are in stopwords, hence the length of list of token in query is 0
            return []
        
        if len(query) == 1 and query[0] not in self.term_id_map:
            return []

        selected_token = []
        for token in query:
            if token not in self.term_id_map:
                continue
            selected_token.append(self.term_id_map[token])
        query = selected_token
        
        scores = {} # key: docid, val: score
        res = []
        with InvertedIndexReader(index_name=self.index_name, postings_encoding=self.postings_encoding, directory=self.output_dir) as reader:
            for token in query:
                # print(f'token: {self.term_id_map[token]}')
                
                N = len(reader.doc_length)
                # print(N)

                postings_lst = reader.get_postings_list(token) # doc_id, tf(t,D)
                # print(postings_lst)

                # Okapi BM25
                # log(N/dft).(((k1+1).tft)/(k1.((1-b)+b.(dl/avgdl))+tft))
                # idf.rsv-norm -> rsv = ((k1+1).tft)/(k1+tft), doc length norm: (1-b)+b.(dl/avgdl)

                dft = len(postings_lst[0]) # total document in collection that contain the term t
                # print(f'dft: {dft}')
                idf = math.log10(N/dft) # idf => log(N/dft)

                if self.avg_doc_length is None:
                    # avg_doc_length: total seluruh doc_length untuk tiap doc_id pada collections/banyak collections
                    total_doc_len = sum(reader.doc_length.values())
                    self.avg_doc_length = total_doc_len/len(reader.doc_length)

                for (docid, tf) in zip(*postings_lst):
                    # print(f'(docid, tf): {(docid, tf)}')

                    rsv_norm = ((k1+1)*tf)/(k1*((1-b)+b*(reader.doc_length[docid]/self.avg_doc_length))+tf)

                    if docid not in scores:
                        scores[docid] = 0
                    scores[docid] += idf*rsv_norm


            top_k = list(heapq.nlargest(k, scores.items(), key=lambda x: x[1])) # get top k highest score document, compare using x[1]: score of each doc
            # print(f'top_k: {top_k}')

            for docid, score in top_k:
                res.append((score, self.doc_id_map[docid]))
        return res

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='main/static/data/collections',
                              postings_encoding=VBEPostings,
                              output_dir='main/static/data/index')
    BSBI_instance.do_indexing()  # memulai indexing!
