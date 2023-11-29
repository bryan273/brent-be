from collections import defaultdict
import os
import pickle
import contextlib
import heapq
import math
import re

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
nltk.data.path.append("./nltk_data")
from nltk.stem import PorterStemmer
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
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

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

    def pre_processing_text(self, content):
        # print(content)
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        words = nltk.word_tokenize(content)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        filtered_words = [word for word in stemmed_words if word.lower() not in self.stop_words]
        preprocessed_text = ' '.join(filtered_words)
        return preprocessed_text

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
        td_pairs = []
        block_full_path = os.path.join(self.data_dir, block_path)
        
        # iterate tiap txt
        for file in os.listdir(block_full_path):
            if file.endswith(".txt"):
                file_path = os.path.join(block_full_path, file)
                # buat doc id
                doc_id = self.doc_id_map[file_path]
                
                # read, process, buat term id, dan append ke td_pairs
                with open(file_path, 'r', encoding='utf-8') as doc_file:
                    text = doc_file.read()
                    tokens = re.findall(r"\w+", text)
                    text_clean = " ".join(tokens)
                    terms = self.pre_processing_text(text_clean).split(' ')
        
                    for term in terms:
                        term_id = self.term_id_map[term]
                        td_pairs.append((term_id, doc_id))
        
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
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id in term_dict:
                postings_list, tf_list, idx_dict = term_dict[term_id]
                # kalo doc id udah ada, tinggal nambahin tf nya
                if doc_id in idx_dict:
                    idx = idx_dict[doc_id]
                    tf_list[idx] += 1
                else:
                    # kalo blom ada, tambahkan doc_id baru dengan tf= 1
                    postings_list.append(doc_id)
                    tf_list.append(1)
                    idx_dict[doc_id] = len(postings_list) - 1  # simpan indeks baru ke dict
            else:
                # inisialisasi term baru kalo blm ada
                term_dict[term_id] = ([doc_id], [1], {doc_id: 0})

        # simpen ke harddisk
        for term_id, (postings_list, tf_list, _) in sorted(term_dict.items(), key=lambda x: x[0]):
            index.append(term_id, postings_list, tf_list)

       
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

        w(t, D) = (1 + log10 tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log10 (N / df(t))

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
        self.load()
        # proses query
        # tokens = re.findall(r"\w+", query)
        # clean_text = " ".join(tokens)
        # terms = self.pre_processing_text(clean_text).split(' ')
        
        terms = query.split() # asumsi proses telah dilakukan 
        scores = defaultdict(int)
        counter = 0
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            # banyak dokumen
            N = len(index.doc_length)
            for term in terms:
                # kalo term gada gausah diproses
                if term in self.term_id_map:
                    term_id = self.term_id_map[term]
                    # ada berapa banyak dokumen yang memiliki term itu
                    # caranya: akses ambil panjang posting list dari index 1
                    df = index.postings_dict[term_id][1]
                    idf = math.log10(N / df)
                    postings, tf_list = index.get_postings_list(term_id)

                    # Itung TF-IDF untuk setiap dokumen dan nanti diliat manaa yang
                    # paling relevan
                    for doc_id, tf in zip(postings, tf_list):
                        doc_name = self.doc_id_map[doc_id]
                        scores[doc_name] += (1 + math.log10(tf)) * idf
                        counter+=1
                else:
                    # query di skip karena tidak ada
                    # raise ValueError("Terms tidak ada di collection")
                    pass

        # sort top k tf idf terbesar
        sorted_scores = sorted(scores.items(), key=itemgetter(1), reverse=True)
        # print("fully evaluated",counter)
        return [(score, doc) for doc, score in sorted_scores[:k]]

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
        self.load()
        tokens = re.findall(r"\w+", query)
        clean_text = " ".join(tokens)
        terms = self.pre_processing_text(clean_text).split(' ')
        scores = defaultdict(int)
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            # banyak dokumen
            N = len(index.doc_length)
            # avgdl = sum(index.doc_length.values()) / N
            # komponen avg.dl load dari inverted index supaya tidak dihitung berkali-kali
            avgdl = index.avgdl
            for term in terms:

                if term in self.term_id_map:
                    term_id = self.term_id_map[term]
                    # calculate idf
                    df = index.postings_dict[term_id][1]
                    idf = math.log10(N / df)

                    postings, tf_list = index.get_postings_list(term_id)
                    for doc_id, tf in zip(postings, tf_list):
                        # itung pake rumus BM25
                        doc_name = self.doc_id_map[doc_id]
                        dl = index.doc_length[doc_id]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (dl / avgdl))
                        scores[doc_name] += idf * (numerator / denominator)
                else:
                    # query di skip karena tidak ada
                    # raise ValueError("Terms tidak ada di collection")
                    pass
        
        # top k retrieval dari skor bm25
        sorted_scores = sorted(scores.items(), key=itemgetter(1), reverse=True)
        return [(score, doc) for doc, score in sorted_scores[:k]]

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
