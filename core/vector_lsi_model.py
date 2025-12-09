import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
import os
import joblib

class VectorLSIModel:

    def __init__(self, text_col: str = "speech", target: float = 0.75):
        self.text_col = text_col
        self.target = target
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df = 5, token_pattern=r'(?u)[^\W\d_]{2,}')
        self.normaliser = Normalizer(copy=False)
        
        #tdf
        self.tdfidf_matrix = None
        self.df = None

        #svd
        self.svd_model = None
        self.cum_var = None
        self.doc_vectors = None
        self.k_auto = None


    def fit(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        docs = self.df[self.text_col].astype(str).tolist()
        #sklearn vectoriser produses a document-term matrix
        self.tdfidf_matrix = self.vectorizer.fit_transform(docs)
        ##TODO : CHANGE TO LOGS
        print(f"TF-IDF matrix shape: {self.tdfidf_matrix.shape}")

    
    def fit_lsi(self) -> None:
        n_docs, n_terms = self.tdfidf_matrix.shape
        max_components = min(n_docs - 1, n_terms - 1) # SVD constraint
        self.svd_model = TruncatedSVD(n_components=max_components, random_state=42)
        docs_vector_full = self.svd_model.fit_transform(self.tdfidf_matrix) #documents in rows, concepts in columns

        self.cum_var = np.cumsum(self.svd_model.explained_variance_ratio_) #cumulative variance explained by each component, how much info i get by adding each new concept

        idx = np.searchsorted(self.cum_var, self.target) #find the index where cumulative variance meets or exceeds target
        if idx>=len(self.cum_var):
            idx = len(self.cum_var) -1
        self.k_auto = idx + 1  # +1 because indices start at 0

        self.doc_vectors = docs_vector_full[:, :self.k_auto] #select only the first k_auto components

        self.doc_vectors = self.normaliser.fit_transform(self.doc_vectors) #normalize the document vectors to unit length, important for cosine similarity, rows are documents, columns are concepts

        print(f"Auto-selected k = {self.k_auto}, cumulative variance = {self.cum_var[self.k_auto-1]:.3f}")
        print(f"Final LSI doc vectors shape: {self.doc_vectors.shape}")

    def query_vector(self, query: str) -> np.array:
        query_tfidf = self.vectorizer.transform([query]) #transform query to tf-idf vector
        quert_lsi_full = self.svd_model.transform(query_tfidf) #project query tf-idf vector to LSI space
        query_lsi = quert_lsi_full[:, :self.k_auto] #select only the first k_auto components
        query_lsi = self.normaliser.transform(query_lsi) #normalize the query vector to unit length
        return query_lsi
    
    def __cosine_simi(self, query_vec: np.array)-> np.ndarray:
        return (self.doc_vectors @ query_vec.T).flatten() #dot product between document vectors and query vector
     

    def save_models(self, folder: str = "models/lsi") -> None:
        os.makedirs(folder, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(folder, "tfidf_vectorizer.joblib"))
        joblib.dump(self.svd_model, os.path.join(folder, "svd_model.joblib"))
        joblib.dump(self.normaliser, os.path.join(folder, "normaliser.joblib"))
        joblib.dump(self.k_auto,  os.path.join(folder, "k_auto.joblib"))

        if self.doc_vectors is not None:
            joblib.dump(self.doc_vectors, os.path.join(folder, "doc_vectors.npy"))

        if self.df is not None:
            self.df.to_csv(os.path.join(folder, "documents.csv"), index=False)

    def load_models(self, folder: str = "models/lsi") -> None:
        self.vectorizer = joblib.load(os.path.join(folder, "tfidf_vectorizer.joblib"))
        self.svd_model = joblib.load(os.path.join(folder, "svd_model.joblib"))
        self.normaliser = joblib.load(os.path.join(folder, "normaliser.joblib"))
        self.k_auto = joblib.load(os.path.join(folder, "k_auto.joblib"))

        self.doc_vectors = joblib.load(os.path.join(folder, "doc_vectors.npy"))
        self.df = pd.read_csv(os.path.join(folder, "documents.csv"))

       

