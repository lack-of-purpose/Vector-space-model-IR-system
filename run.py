import gensim
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import ufal.morphodita as morph
import string
import argparse
import numpy as np
from bs4 import BeautifulSoup
import scipy.sparse as sp
from collections import OrderedDict
import re
        
        
def tokenize(input):
    #whitespace+punctuation
    #(data reading, tokenization, punctuation removal, …)
    if TOKENIZER == None:
        en = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        cs = 'aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHCIÍJKLMNŇÓPQRŘSŤUÚŮVWXYÝZŽ'
        output = []
        words = []
        word = ''
        output = re.findall(r"[a-zA-Z]+", input)
        if ' ' in output:
            output.remove(' ')
        '''
        for symbol in input:
            if LANGUAGE == 'en':
                if symbol in en:
                    word += symbol
                else:
                    if word != '':
                        words.append(word)
                    word = ''
            else:
                if symbol in cs:
                    word += symbol
                else:
                    if word != '':
                        words.append(word)
                    word = ''
        for word in words:
                output.append(word)     
        '''
        return output
    elif TOKENIZER == 'nltk':
        if LANGUAGE == 'cz':
            return word_tokenize(input, language='czech')
        else:
            return word_tokenize(input)
    elif TOKENIZER == 'gensim':
        return list(gensim.utils.tokenize(input))
        
    
def class_equivalence():
    #no
    #(case folding, stemming, lemmatization, number normalization, …)
    ...
    
def stopwords_remove(input):
    #no
    #(none, frequency/POS/lexicon-based)
    if STOPWORDS:
        en_stop_words = set(stopwords.words('english'))
        output = [w for w in input if not w in en_stop_words]
        return output
    
def query_construction(queries):
    #all words from ”title”
    #(automatic, manual)
    queries_file = open(queries).read()
    parseObj = BeautifulSoup(queries_file, features="xml")
    if len(QUERY) == 1: 
        titles = parseObj.find_all(QUERY[0])
        list_of_queries = []
        for item in titles:
            query = tokenize(item.next)
            list_of_queries.append(query)
    if len(QUERY) == 2:
        titles = parseObj.find_all(QUERY[0])
        additional1 = parseObj.find_all(QUERY[1])
        list_of_queries = []
        for item1, item2 in zip(titles, additional1):
            query = tokenize(item1.next) + tokenize(item2.next)
            list_of_queries.append(query)
    if len(QUERY) == 3:
        titles = parseObj.find_all(QUERY[0])
        additional1 = parseObj.find_all(QUERY[1])
        additional2 = parseObj.find_all(QUERY[2])
        list_of_queries = []
        for item1, item2, item3 in zip(titles, additional1, additional2):
            query = tokenize(item1.next) + tokenize(item2.next) + tokenize(item3.next)
            list_of_queries.append(query)
        
def df(input: np.ndarray, num_of_docs: int):
    #none
    #(none, idf, probabilistic idf )
    if DF_TYPE == None:
        return np.ones(input.shape)
    elif DF_TYPE == 'idf':
        return np.log(num_of_docs/input)
    elif DF_TYPE == 'prob':
        df = np.log((num_of_docs - input)/input)
        zeros = np.zeros(df.shape)
        return np.maximum(zeros, df)

def vector_norm(vec: np.ndarray):
    #cosine
    #(none, cosine, pivoted)
    vec = vec/np.sum(vec**2)
    return vec

def matrix_norm(matrix: np.ndarray):
    row_sum = np.sum(matrix**2, axis=1)
    return matrix/row_sum[:, np.newaxis]

def write_to_file(file, list_of_relevants, query_id, scores):
    #10.2452/401-AH 0 LN-20020201065 0 0.9 baseline  qid, iter, docno, rank, sim, run_id
    for i, doc_no, score in zip(enumerate(list_of_relevants), scores):
        file.write(f'{query_id} 0 {doc_no} {i} {score} baseline\n')
        
    
def similarity(query_matrix: np.ndarray, doc_matrix: np.ndarray, docs_dict, queries_dict, filename):
    #cosine
    #(cosine, BM25)
    result_file = open(filename, 'w')
    for query in range(query_matrix.shape[0]):
        query_num = queries_dict[query]
        sim = np.sum(query_matrix[query, :] * doc_matrix, axis=1) / (np.sum(query_matrix[query, :]**2) * np.sum(doc_matrix**2, axis=1))
        sorted_sim = np.argsort(sim)  #indices of sorted similarity scores
        desc = np.flip(sorted_sim)  #reverse it to descending order
        top_ranked = sim[desc[:1000]] #ranks of first 1000 most relevant
        num_of_relevant = (top_ranked > 0).sum() #amount of relevant docs (sim > 0)
        nums_of_rel_docs = desc[:num_of_relevant] #numbers of that docs
        list_of_relevant_doc_no = [docs_dict[id] for id in nums_of_rel_docs]
        ranks_of_relevant_docs = list(top_ranked[:num_of_relevant])
        write_to_file(result_file, list_of_relevant_doc_no, query_num, ranks_of_relevant_docs)

def tf_matrix(list_of_terms, dict_of_docs, term_dict):
    tf_dict = {}
    for doc_id in dict_of_docs.keys():
        if not doc_id in tf_dict.keys():
            tf_dict[doc_id] = [0] * len(list_of_terms)
        for word in dict_of_docs[doc_id]:
            if word in term_dict.keys():
                tf_dict[doc_id][term_dict[word]] += 1
        #for term in term_dict.keys():
        #    if term in dict_of_docs[doc_id]:
        #        tf_dict[doc_id][term_dict[term]] += 1
    doc_tf_matrix = np.array(list(tf_dict.values()))
    if TF_TYPE == 'logarithm':
        doc_tf_matrix = 1 + np.log(doc_tf_matrix)
    elif TF_TYPE == 'augmented':
        doc_tf_matrix = 0.5 + 0.5*doc_tf_matrix/np.max(doc_tf_matrix)
    return doc_tf_matrix
                
def fill_df_dict(df_list, dict_of_docs, term_dict):
    df_list = [0] * len(term_dict.keys())
    for doc_id in dict_of_docs.keys():
        for term in dict_of_docs[doc_id]:
            df_list[term_dict[term]] += 1
    
    return df_list
            
def doc_tf_idf(tf_matrix, df_vector, N):
    if DF_TYPE == 'idf':
        idf = np.log(N/df_vector)
    elif DF_TYPE == 'prob':
        df = np.log((N - df_vector)/df_vector)
        zeros = np.zeros(df.shape)
        idf = np.maximum(zeros, df)
    else:
        idf = df_vector

    return matrix_norm(np.multiply(tf_matrix, idf))
    

                 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queries', help='List of queries')
    parser.add_argument('-d', '--docs', help='List of docs')
    parser.add_argument('-r', '--run_id', help='Run id')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-s', '--system_type', help='Type of the system: baseline or None')
    parser.add_argument('-t', '--tokenizer', default=None, help='Tokenizer to use: nltk, spacy, gensim')
    parser.add_argument('-sr', '--stopwords_removal', default=False, help='True or False')
    parser.add_argument('-tf', '--tf_type', default=None, help='logarithm, augmented')
    parser.add_argument('-df', '--df_type', default=None, help='idf, prob')
    parser.add_argument('-qc', '--query_construction', default=['title'], help='List of tags from which to construct a query')
    args = parser.parse_args()
    
    # params of run 
    LANGUAGE = args.queries.split('.')[0].split('_')[1]
    TOKENIZER = args.tokenizer
    STOPWORDS = args.stopwords_removal
    TF_TYPE = args.tf_type
    DF_TYPE = args.df_type
    QUERY = args.query_construction
    
    # vars
    docs_df_list = []
    q_df_list = []
    vocab = set()
    collection_size = 0
    docs_dict = {}
    queries_dict = {}
    
    # parse docs
    docs_list = open(f'input/{args.docs}').readlines()
    all_docs = {}
    for doc_file in docs_list:
        filename = f'documents_{LANGUAGE}/{doc_file.strip()}'
        file = open(filename)
        docs = file.read()
        parseDoc = BeautifulSoup(docs, features="xml")
        temp = parseDoc.find_all(['DOCNO', 'HD', 'LD', 'TE', 'DH', 'CP'])
        current_doc = ''
        for item in temp:
            if item.name == 'DOCNO':
                collection_size += 1
                if current_doc != '':
                    doc = tokenize(all_docs[current_doc])
                    vocab.update(doc)
                    all_docs[current_doc] = doc
                all_docs[item.next] = ''
                current_doc = item.next
                continue
            else:
                all_docs[current_doc] += f'{item.next.strip()} '
        doc = tokenize(all_docs[current_doc])
        vocab.update(doc)
        all_docs[current_doc] = doc
                
    # sort dict and create doc_no : id dict
    all_docs = OrderedDict(sorted(all_docs.items()))
    docs_dict = {id: doc_no for id, doc_no in enumerate(list(all_docs.keys()))}
    
    # create dict of terms : id from sorted list
    list_of_terms = list(vocab)
    term_dict = {term: id for id, term in enumerate(sorted(list_of_terms))}
    
    # compute tf and df based on docs
    doc_tf_matrix = tf_matrix(list_of_terms, all_docs, term_dict)
    df_list = fill_df_dict(docs_df_list, all_docs, term_dict)
    doc_df_vector = np.array(df_list)
    
    #compute tf_idf for docs
    doc_tf_idf_matrix = doc_tf_idf(doc_tf_matrix, doc_df_vector, collection_size)

    
    # parse queries
    queries_file = open(args.queries).read()
    parseObj = BeautifulSoup(queries_file, features="xml")
    queries = parseObj.find_all(['num', 'title'])
    all_queries = {}
    for item in queries:
        if item.name == 'num':
            num = item.next
            continue
        query = tokenize(item.next)
        all_queries[num] = query
    all_queries = OrderedDict(sorted(all_queries.items()))
    
    # create dict query :  id
    queries_dict = {num: query for num, query in enumerate(list(all_queries.keys()))}
    
    q_tf_matrix = tf_matrix(list_of_terms, all_queries, term_dict)
    q_tf_idf_matrix = doc_tf_idf(q_tf_matrix, doc_df_vector, collection_size)
    
    
    similarity(q_tf_idf_matrix, doc_tf_idf_matrix, docs_dict, queries_dict, args.output)