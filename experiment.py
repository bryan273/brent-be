import re
import os
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
import math
# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    # DCG(p) = sigma(i=1 until p) [rel_i/(log_2_(i+1))]
    score = 0.0
    for i in range(len(ranking)):
        rel_i = ranking[i]
        # kalo rel_i = 0, maka gausah diitung
        if rel_i == 1:
            score += 1.0 / (math.log2(i + 2))  # i+2 karena index mulai dari 0
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    # Prec@K = Number of relevant documents in top K / K
    relevant_docs = sum(ranking[:k])
    return relevant_docs / k


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    # AP= (1/R) ​∑(k=1 until N) [Prec@k * rel_k]
    # r adalah jumlah dokumen yang relevan
    R = sum(ranking)
    if R == 0:  # menghindari div by 0 kalo gada yg relevan dokumennya
        return 0
    prec_at_k_list = [prec(ranking, k + 1) * ranking[k] for k in range(len(ranking))]
    return sum(prec_at_k_list) / R

# >>>>> memuat qrels


def load_qrels(qrel_file="qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    with open(qrel_file) as file:
        content = file.readlines()

    qrels_sparse = {}

    for line in content:
        parts = line.strip().split()
        qid = parts[0]
        did = int(parts[1])
        if not (qid in qrels_sparse):
            qrels_sparse[qid] = {}
        if not (did in qrels_sparse[qid]):
            qrels_sparse[qid][did] = 0
        qrels_sparse[qid][did] = 1
    return qrels_sparse

# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="queries.txt", k=1000):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    with open(query_file) as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []

        rbp_scores_bm25_1 = []
        dcg_scores_bm25_1 = []
        ap_scores_bm25_1 = []

        rbp_scores_bm25_2 = []
        dcg_scores_bm25_2 = []
        ap_scores_bm25_2 = []

        rbp_scores_bm25_3 = []
        dcg_scores_bm25_3 = []
        ap_scores_bm25_3 = []

        rbp_scores_bm25_4 = []
        dcg_scores_bm25_4 = []
        ap_scores_bm25_4 = []

        rbp_scores_bm25_5 = []
        dcg_scores_bm25_5 = []
        ap_scores_bm25_5 = []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi TF-IDF
            """
            ranking_tfidf = []
            for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
                # print(doc)
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)

            # print(sum(ranking_tfidf))
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            Evaluasi BM25 k1=1.2 dan b=0.75
            """
            ranking_bm25_1 = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=1.2, b=0.75):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_bm25_1.append(1)
                else:
                    ranking_bm25_1.append(0)
            rbp_scores_bm25_1.append(rbp(ranking_bm25_1))
            dcg_scores_bm25_1.append(dcg(ranking_bm25_1))
            ap_scores_bm25_1.append(ap(ranking_bm25_1))

            """
            Evaluasi BM25 k1=2 dan b=0.75
            """
            ranking_bm25_2 = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=2, b=0.75):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_bm25_2.append(1)
                else:
                    ranking_bm25_2.append(0)
            rbp_scores_bm25_2.append(rbp(ranking_bm25_2))
            dcg_scores_bm25_2.append(dcg(ranking_bm25_2))
            ap_scores_bm25_2.append(ap(ranking_bm25_2))

            """
            Evaluasi BM25 k1=1.2 dan b=0.9
            """
            ranking_bm25_3 = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=1.2, b=0.9):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_bm25_3.append(1)
                else:
                    ranking_bm25_3.append(0)
            rbp_scores_bm25_3.append(rbp(ranking_bm25_3))
            dcg_scores_bm25_3.append(dcg(ranking_bm25_3))
            ap_scores_bm25_3.append(ap(ranking_bm25_3))

            """
            Evaluasi BM25 k1=0.6 dan b=0.6
            """
            ranking_bm25_4 = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=0.6, b=0.6):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_bm25_4.append(1)
                else:
                    ranking_bm25_4.append(0)
            rbp_scores_bm25_4.append(rbp(ranking_bm25_4))
            dcg_scores_bm25_4.append(dcg(ranking_bm25_4))
            ap_scores_bm25_4.append(ap(ranking_bm25_4))

            """
            Evaluasi BM25 k1=3 dan b=3
            """
            ranking_bm25_5 = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1=3, b=3):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_bm25_5.append(1)
                else:
                    ranking_bm25_5.append(0)
            rbp_scores_bm25_5.append(rbp(ranking_bm25_5))
            dcg_scores_bm25_5.append(dcg(ranking_bm25_5))
            ap_scores_bm25_5.append(ap(ranking_bm25_5))

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))

    print("Hasil evaluasi BM25 k=1.2 dan b=0.75 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_1) / len(rbp_scores_bm25_1))
    print("DCG score =", sum(dcg_scores_bm25_1) / len(dcg_scores_bm25_1))
    print("AP score  =", sum(ap_scores_bm25_1) / len(ap_scores_bm25_1))

    print("Hasil evaluasi BM25 k=2 dan b=0.75 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_2) / len(rbp_scores_bm25_2))
    print("DCG score =", sum(dcg_scores_bm25_2) / len(dcg_scores_bm25_2))
    print("AP score  =", sum(ap_scores_bm25_2) / len(ap_scores_bm25_2))

    print("Hasil evaluasi BM25 k=1.2 dan b=0.9 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_3) / len(rbp_scores_bm25_3))
    print("DCG score =", sum(dcg_scores_bm25_3) / len(dcg_scores_bm25_3))
    print("AP score  =", sum(ap_scores_bm25_3) / len(ap_scores_bm25_3))

    print("Hasil evaluasi BM25 k=0.6 dan b=0.6 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_4) / len(rbp_scores_bm25_4))
    print("DCG score =", sum(dcg_scores_bm25_4) / len(dcg_scores_bm25_4))
    print("AP score  =", sum(ap_scores_bm25_4) / len(ap_scores_bm25_4))

    print("Hasil evaluasi BM25 k=3 dan b=3 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_5) / len(rbp_scores_bm25_5))
    print("DCG score =", sum(dcg_scores_bm25_5) / len(dcg_scores_bm25_5))
    print("AP score  =", sum(ap_scores_bm25_5) / len(ap_scores_bm25_5))


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1002252"][5599474] == 1, "qrels salah"
    assert not (6998091 in qrels["Q1007972"]), "qrels salah"

    eval_retrieval(qrels)
