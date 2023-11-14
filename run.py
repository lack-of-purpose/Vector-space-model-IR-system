import os
import gensim
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import ufal.morphodita as morph
import string
import argparse
import numpy as np
from bs4 import BeautifulSoup

class IR_system():
    
    def __init__(self, spec, list_of_queries, all_docs):
        self.specification = spec[0]
        if spec[0] == 'baseline':
            self.tokenizer = None
            self.stopwords = False
            self.tf_type = None
            self.df_type = None
        self.tokenizer = spec[1]
        self.lang = spec[2]
        self.stopwords = spec[3]
        self.tf_type = spec[4]
        self.df_type = spec[5]
        self.queries = list_of_queries
        self.docs = all_docs
        #if spec[2] == 'en':
        #    self.nlp = spacy.load('en')
        
        
    def tokenization(self, input):
        #whitespace+punctuation
        #(data reading, tokenization, punctuation removal, …)
        if self.tokenizer == None:
            output = []
            for word in input:
                if not word in list(string.punctuation):
                    output.append(word)
            return output
        elif self.tokenizer == 'nltk':
            if self.lang == 'cz':
                return word_tokenize(input, language='czech')
            else:
                return word_tokenize(input)
        elif self.tokenizer == 'spacy':
            doc = self.nlp(input)
            output = []
            for token in doc:
                output.append(token)
            return output
        else:
            return list(gensim.utils.tokenize(input))
        
    def class_equivalence():
        #no
        #(case folding, stemming, lemmatization, number normalization, …)
        ...
        
    def stopwords_remove(self, input):
        #no
        #(none, frequency/POS/lexicon-based)
        if self.stopwords:
            en_stop_words = set(stopwords.words('english'))
            output = [w for w in input if not w in en_stop_words]
            return output
    
    def text_preprocessing(self, text):
        ...
        
    def query_construction():
        #all words from ”title”
        #(automatic, manual)
        ...
        
    def tf(self, input):
        #natural
        #(boolean, natural, logarithm, log average, augmented)
        if self.tf_type == None:
            return input
        elif self.tf_type == 'logarithm':
            return np.log(input) + 1
        elif self.tf_type == 'augmented':
            return 0.5 + 0.5*input/np.max(input)
            
    
    def df(self, input, num_of_docs):
        #none
        #(none, idf, probabilistic idf )
        if self.df_type == None:
            return np.ones(input.shape)
        elif self.df_type == 'idf':
            return np.log(num_of_docs/input)
        elif self.df_type == 'prob':
            df = np.log((num_of_docs - input)/input)
            zeros = np.zeros(df.shape)
            return np.maximum(zeros, df)
    
    def vector_norm(self, vec: np.ndarray):
        #cosine
        #(none, cosine, pivoted)
        vec = vec/np.sum(vec**2)
        
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray):
        #cosine
        #(cosine, BM25)
        return (vec1 + vec2)/np.sum(vec1**2) * np.sum(vec2**2)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queries', help='List of queries')
    parser.add_argument('-d', '--docs', help='List of docs')
    parser.add_argument('-r', '--run_id', help='Run id')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-s', '--system-type', help='Type of the system: baseline or None')
    parser.add_argument('-t', '--tokenizer', default=None, help='Tokenizer to use: nltk, spacy, gensim')
    parser.add_argument('-sr', '--stopwords-removal', default=False, help='True or False')
    parser.add_argument('-tf', '--tf-type', default=None, help='logarithm, augmented')
    parser.add_argument('-df', '--df-type', default=None, help='idf, prob')
    args = parser.parse_args()
    
    language = args.queries.split('.')[0].split('_')[1]
    #specification: type of system, tokenizer, lang, stopwords removal, tf_type, df_type 
    spec = ['baseline', None, language, True, None, None]
    
    system = IR_system(spec)
    
    queries_file = open(args.queries).read()
    parseObj = BeautifulSoup(queries_file, features="xml")
    titles = parseObj.find_all('title')
    list_of_queries = []
    for item in titles:
        #start = item.find('<title>')
        #end = item.find('</title>')
        list_of_queries.append(item.next)#(item[start + len('<title>') + 1:end])
    
    
    docs_list = open(f'input/{args.docs}').readlines()
    all_docs = {}
    for doc_file in docs_list:
        filename = f'documents_{language}/{doc_file.strip()}'
        file = open(filename)
        docs = file.read()
        #doc_nums = re.findall(r'\<DOCNO>(.*?)\</DOCNO>', docs)
        parseDoc = BeautifulSoup(docs, features="xml")
        temp = parseDoc.find_all(['DOCNO', 'HD', 'LD', 'TE', 'DH', 'CP'])
        current_doc = ''
        for item in temp:
            if item.name == 'DOCNO':
                all_docs[item.next] = ''
                current_doc = item.next
                continue
            else:
                all_docs[current_doc] += f'{item.next} '
                
    
    system = IR_system(spec, list_of_queries, all_docs)