import gensim
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import ufal.morphodita as morph
import string
import sys
import argparse
import numpy as np
from bs4 import BeautifulSoup
import scipy.sparse as sp
from collections import OrderedDict
import re
import time
from sklearn.preprocessing import normalize
import math
        
        
def tokenize(input):
    #whitespace+punctuation
    #(data reading, tokenization, punctuation removal, …)
    if TOKENIZER == None:
        en = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        cs = 'aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHCIÍJKLMNŇÓPQRŘSŤUÚŮVWXYÝZŽ'
        output = []
        words = []
        word = ''
        if LANGUAGE == 'en':
            output = re.findall(r"[a-zA-Z]+", input)
        else:
            output = re.findall(r"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHCIÍJKLMNŇÓPQRŘSŤUÚŮVWXYÝZŽ]+", input)
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
    queries_file = open(queries, encoding='utf-8').read()
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

def write_to_file(file, list_of_relevants, query_id, scores):
    #10.2452/401-AH 0 LN-20020201065 0 0.9 baseline  qid, iter, docno, rank, sim, run_id
    for i, doc_no, score in zip(range(len(list_of_relevants)),list_of_relevants, scores):
        file.write(f'{query_id} 0 {doc_no} {i} {score} baseline\n')
        
    
def similarity(query_matrix: np.ndarray, doc_matrix: np.ndarray, docs_dict, queries_dict, filename):
    #cosine
    #(cosine, BM25)
    result_file = open(filename, 'w', encoding='utf-8')
    for query in range(query_matrix.shape[0]):
        query_num = queries_dict[query]
        divident = np.sum(query_matrix[[query], :] * doc_matrix, axis=1)
        divisor = (np.sum(query_matrix[[query], :]**2) * np.sum(doc_matrix**2, axis=1))
        sim = np.divide(divident, divisor)#, out=np.zeros_like(divident), where=divisor!=0)
        sim[np.isnan(sim)] = 0
        #np.nan_to_num(sim, nan=0)
        sorted_sim = np.argsort(sim)  #indices of sorted similarity scores
        desc = np.flip(sorted_sim)  #reverse it to descending order
        top_ranked = sim[desc[:1000]] #ranks of first 1000 most relevant
        num_of_relevant = (top_ranked > 0).sum() #amount of relevant docs (sim > 0)
        nums_of_rel_docs = desc[:num_of_relevant] #numbers of that docs
        list_of_relevant_doc_no = [docs_dict[id] for id in nums_of_rel_docs]
        ranks_of_relevant_docs = list(top_ranked[:num_of_relevant])
        write_to_file(result_file, list_of_relevant_doc_no, query_num, ranks_of_relevant_docs)

def tf_weighting(tf_sparse_matrix):
    if TF_TYPE == None:
        doc_tf_matrix = tf_sparse_matrix
    if TF_TYPE == 'logarithm':
        doc_tf_matrix = 1 + np.log(tf_sparse_matrix)
    elif TF_TYPE == 'augmented':
        doc_tf_matrix = 0.5 + 0.5*tf_sparse_matrix/np.max(tf_sparse_matrix, axis=0)
    return doc_tf_matrix

def update_df(df_list, word_set, term_dict):
    words = list(word_set)
    for word in words:
        df_list[term_dict[word]] += 1
            
    return df_list

def update_tf(tf_dict, doc_num, doc):
    tf_dict[doc_num] = {}
    for word in doc:
        if not word in tf_dict[doc_num].keys():
            tf_dict[doc_num][word] = 1
        else:
            tf_dict[doc_num][word] += 1
    
    return  tf_dict
            
def doc_tf_idf(tf_matrix, df_vector, N):
    if DF_TYPE == 'idf':
        idf = np.log(N/df_vector)
    elif DF_TYPE == 'prob':
        df = np.log((N - df_vector)/df_vector)
        zeros = np.zeros(df.shape)
        idf = np.maximum(zeros, df)
    else:
        idf = np.ones(df_vector.shape[0]) #df_vector
    
    tf_idf = tf_matrix*idf#sp.csr_matrix.dot(idf, tf_matrix)
    return normalize(tf_idf, norm='l2')

def get_doc_text(doc):
    if LANGUAGE == 'en':
        doc_id = re.findall(r'<DOCNO>(.*?)</DOCNO>', doc)[0]
        doc_inside = re.sub(r'<DOC(ID|NO)>(.*?)</DOC(ID|NO)>\n', '', doc)
        useful_tags = re.findall(r'<(HD|LD|TE|DH|CP)>((.|\n)*?)</(HD|LD|TE|DH|CP)>', doc_inside)
    if LANGUAGE == 'cs':
        doc_id = re.findall(r'<DOCNO>(.*?)</DOCNO>', doc)[0]
        doc_inside = re.sub(r'<DOC(ID|NO)>(.*?)</DOC(ID|NO)>\n', '', doc)
        useful_tags = re.findall(r'<(TITLE|HEADING|GEOGRAPHY|TEXT)>((.|\n)*?)</(TITLE|HEADING|GEOGRAPHY|TEXT)>', doc_inside)
    current_doc = ''
    for tag in useful_tags:
        current_doc += tag[1]
    
    return doc_id, current_doc
    
    
def parse_xml(filename, collection_size, list_of_docs, vocab):
    file = open(filename, encoding='utf-8')
    f = file.read()
    all_docs = re.findall(r'<DOC>((.|\n)*?)</DOC>', f)
    docs = [doc[0] for doc in all_docs]
    for doc in docs:
        doc_id, current_doc = get_doc_text(doc)
        doc = tokenize(current_doc)
        vocab.update(doc)
        collection_size += 1
        list_of_docs.append(doc_id)
    file.close()
    return vocab, list_of_docs, collection_size
    
def parse_xml_with_update(filename, df_list, tf_sparse_matrix, term_dict, docs_dict):
    file = open(filename, encoding='utf-8')
    f = file.read()
    all_docs = re.findall(r'<DOC>((.|\n)*?)</DOC>', f)
    docs = [doc[0] for doc in all_docs]
    for doc in docs:
        doc_id, current_doc = get_doc_text(doc)
        doc = tokenize(current_doc)
        words_set = set(doc)
        df_list = update_df(df_list, words_set, term_dict)
        local_tf = {}
        for word in doc:
            if not word in local_tf.keys():
                local_tf[word] = 1
            else:
                local_tf[word] += 1
        for word in local_tf.keys():
            current_col = term_dict[word]
            current_row = docs_dict[doc_id]
            if TF_TYPE == 'logarithm':
                tf_sparse_matrix[current_row, current_col] = 1 + math.log(local_tf[word])
            elif TF_TYPE == 'boolean':
                tf_sparse_matrix[current_row, current_col] = 1
            else:
                tf_sparse_matrix[current_row, current_col] = local_tf[word]
            
    return tf_sparse_matrix, df_list
                 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queries', help='List of queries')
    parser.add_argument('-d', '--docs', help='List of docs')
    parser.add_argument('-r', '--run_id', help='Run id')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-s', '--system_type', help='Type of the system: baseline or None')
    parser.add_argument('-t', '--tokenizer', default=None, help='Tokenizer to use: nltk, spacy, gensim')
    parser.add_argument('-sr', '--stopwords_removal', default=False, help='True or False')
    parser.add_argument('-tf', '--tf_type', default=None, help='logarithm, boolean')
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
    vocab = set()
    collection_size = 0
    
    tf_dict = {} # {doc_id: {term1: num1, term2: num2, ...}}
    list_of_docs = []
    # parse docs
    start_parse_docs = time.time()
    docs_list = open(f'input/{args.docs}', encoding='utf-8').readlines()
    for doc_file in docs_list:
        filename = f'documents_{LANGUAGE}/{doc_file.strip()}'
        vocab, list_of_docs, collection_size = parse_xml(filename, collection_size, list_of_docs, vocab)  
    # create dict of terms : id from sorted list
    list_of_terms = list(vocab)
    term_dict = {term: id for id, term in enumerate(sorted(list_of_terms))}
    
    i = len(list_of_docs)
    j = len(list_of_terms)
    # create dict of docs : id from sorted list
    docs_dict = {doc: id for id, doc in enumerate(sorted(list_of_docs))}
    df_list = [0] * len(list_of_terms)
    tf_sparse_matrix = sp.dok_array((i, j), dtype=np.int64)
    for doc_file in docs_list:
        filename = f'documents_{LANGUAGE}/{doc_file.strip()}'
        tf_sparse_matrix, df_list = parse_xml_with_update(filename, df_list, tf_sparse_matrix, term_dict, docs_dict)
    docs_dict = {id: doc for id, doc in enumerate(sorted(list_of_docs))}
    #tf_sparse_matrix = sp.csr_array((tf, (row, col)), shape=(i, j))
    end_parse_docs = time.time()
    print(f'parse docs:{end_parse_docs-start_parse_docs}')
    # sort dict and create doc_no : id dict
    #tf_ordered = OrderedDict(sorted(tf_dict.items()))
    #docs_dict = {id: doc_no for id, doc_no in enumerate(list_of_docs)}
    #df_ordered = sorted(df_list)
    doc_df_vector = np.array(df_list).astype('int64')
    doc_tf_matrix = tf_sparse_matrix.tocsr()
    #doc_tf_matrix = tf_weighting(doc_tf_matrix)
    doc_tf_idf_matrix = doc_tf_idf(doc_tf_matrix, doc_df_vector, collection_size)
    
    #list_of_terms = list(vocab)
    #term_dict = {term: id for id, term in enumerate(sorted(list_of_terms))}
    
    # parse queries
    query_list = []
    queries_file = open(args.queries, encoding='utf-8').read()
    if LANGUAGE == 'en':
        all_queries = re.findall(r'<top lang="en">((.|\n)*?)</top>', queries_file)
    else:
        all_queries = re.findall(r'<top lang="cs">((.|\n)*?)</top>', queries_file)
    queries = [query[0] for query in all_queries]
    i = len(queries)
    #query_dict = {query: id for id, query in enumerate(sorted(queries))}
    q_tf_sparse_matrix = sp.dok_array((i, j), dtype=np.int64)
    current_row = 0
    for query in queries:
        query_id = re.findall(r'<num>(.*?)</num>', query)[0]
        query_list.append(query_id)
        query_title = re.findall(r'<title>(.*?)</title>\n', query)[0]
        query_text = tokenize(query_title)
        tf_vec = [0] * len(list_of_terms)
        local_tf = {}
        for word in query_text:
            if not word in local_tf.keys():
                local_tf[word] = 1
            else:
                local_tf[word] += 1
        for word in local_tf.keys():
            if not word in term_dict.keys():
                continue
            current_col = term_dict[word]
            #current_row = query_dict[query_id]
            q_tf_sparse_matrix[current_row, current_col] = local_tf[word]
        current_row += 1
    
    # create dict query :  id
    queries_dict = {num: query for num, query in enumerate(query_list)}
    q_tf_matrix = q_tf_sparse_matrix.tocsr()
    q_tf_idf_matrix = doc_tf_idf(q_tf_matrix, doc_df_vector, collection_size)
    # convert tf to matrix and df to vector
    #start_doc_tf_matrix = time.time()
    #doc_tf_matrix = tf_matrix(tf_dict, list_of_terms, term_dict, True)
    #count = (doc_tf_matrix < 0).sum()
    #end_doc_tf_matrix = time.time()
    #print(f'doc tf compute:{end_doc_tf_matrix-start_doc_tf_matrix}')
    
    #count = (doc_df_vector < 0).sum()
    #q_tf_matrix = tf_matrix(q_tf_dict, list_of_terms, term_dict, False)
    
    #compute tf_idf
    #start_doc_tf_idf_matrix = time.time()
    #doc_tf_idf_matrix = doc_tf_idf(doc_tf_matrix, doc_df_vector, collection_size)
    #end_doc_tf_idf_matrix = time.time()
    #print(f'doc tf idf compute:{end_doc_tf_idf_matrix-start_doc_tf_idf_matrix}')
    
    
    
    start_similarity = time.time()
    similarity(q_tf_idf_matrix, doc_tf_idf_matrix, docs_dict, queries_dict, args.output)
    end_similarity = time.time()
    print(f'similarity: {end_similarity-start_similarity}')