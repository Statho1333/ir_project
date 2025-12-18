import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
import os
import joblib

class VectorLSIModel:

    def __init__(self, text_col: str = "speech", target: float = 0.75):
        """
        Initialize the LSI (Latent Semantic Indexing) model.
        Args:
            text_col (str, optional): The name of the column containing text data to be vectorized.
                Defaults to "speech".
            target (float, optional): The target cumulative explained variance ratio for determining
                the optimal number of components in SVD. Defaults to 0.75.
        Attributes:
            text_col (str): The name of the text column.
            target (float): Target cumulative variance threshold.
            vectorizer (TfidfVectorizer): TF-IDF vectorizer configured with custom parameters
                (max_df=0.95, min_df=5, token_pattern for words with 2+ characters).
            normaliser (Normalizer): L2 normalizer for document vectors (copy=False).
            tfidf_matrix (None or sparse matrix): Stores the TF-IDF matrix. Initialized as None.
            df (None or DataFrame): Stores the input dataframe. Initialized as None.
            svd_model (None or TruncatedSVD): Stores the fitted SVD model. Initialized as None.
            cum_var (None or array-like): Stores cumulative explained variance ratios. Initialized as None.
            doc_vectors (None or ndarray): Stores transformed document vectors. Initialized as None.
            k_auto (None or int): Stores the automatically determined optimal number of components. Initialized as None.
        """
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
        """
        Fit the LSI model on the provided DataFrame.
        
        Extracts text documents from the specified column, transforms them into a 
        TF-IDF matrix using the vectorizer, and stores the resulting sparse matrix.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing the text documents.
        
        Returns:
            None
        
        Notes:
            - Resets the DataFrame index before processing.
            - Converts text column values to strings.
            - Stores the TF-IDF matrix in self.tdfidf_matrix.
            - Prints the shape of the resulting TF-IDF matrix.
        """
        self.df = df.reset_index(drop=True)
        docs = self.df[self.text_col].astype(str).tolist()
        #sklearn vectoriser produses a document-term matrix
        self.tdfidf_matrix = self.vectorizer.fit_transform(docs)
        ##TODO : CHANGE TO LOGS
        print(f"TF-IDF matrix shape: {self.tdfidf_matrix.shape}")

    
    def fit_lsi(self, k_cap: int = 300) -> None:
        """
        Fit a Latent Semantic Indexing (LSI) model using Truncated SVD on the TF-IDF matrix.
        This method performs dimensionality reduction on the TF-IDF matrix and automatically
        selects the number of components based on a target cumulative explained variance threshold.
        The resulting document vectors are normalized.
        Args:
            k_cap (int, optional): Maximum number of components to use in SVD. Defaults to 300.
                The actual number of components is limited by min(k_cap, n_docs - 1, n_terms - 1).
        Raises:
            RuntimeError: If the TF-IDF matrix has not been computed yet (fit() must be called first).
        Attributes set:
            svd_model (TruncatedSVD): The fitted SVD model.
            cum_var (ndarray): Cumulative explained variance ratio for each component.
            k_auto (int): Auto-selected number of components based on target variance threshold.
            doc_vectors (ndarray): Normalized document vectors in the LSI concept space
                with shape (n_docs, k_auto).
        Prints:
            - Initial LSI parameters (n_docs, n_terms, max_components)
            - Auto-selected k value and corresponding cumulative variance
            - Final LSI document vectors shape
        """
        
        if self.tdfidf_matrix is None:
            raise RuntimeError("Call fit() before fit_lsi().")

        n_docs, n_terms = self.tdfidf_matrix.shape
        max_components = min(k_cap, n_docs - 1, n_terms - 1)
        print(f"[LSI] n_docs={n_docs}, n_terms={n_terms}, using {max_components} components")

        self.svd_model = TruncatedSVD(n_components=max_components, random_state=42)
        docs_vector_full = self.svd_model.fit_transform(self.tdfidf_matrix)  # docs x concepts

        self.cum_var = np.cumsum(self.svd_model.explained_variance_ratio_)
        idx = np.searchsorted(self.cum_var, self.target)
        if idx >= len(self.cum_var):
            idx = len(self.cum_var) - 1
        self.k_auto = idx + 1  # indices start at 0

        self.doc_vectors = docs_vector_full[:, :self.k_auto]
        self.doc_vectors = self.normaliser.fit_transform(self.doc_vectors)

        print(f"Auto-selected k = {self.k_auto}, cumulative variance = {self.cum_var[self.k_auto-1]:.3f}")
        print(f"Final LSI doc vectors shape: {self.doc_vectors.shape}")


    def query_vector(self, query: str) -> np.array:
        """
        Transform a query string into a normalized LSI vector representation.
        
        This method converts a raw query string into a Latent Semantic Indexing (LSI)
        vector by applying TF-IDF vectorization, SVD projection, dimensionality reduction,
        and L2 normalization.
        
        Args:
            query (str): The input query string to be transformed.
        
        Returns:
            np.array: A normalized LSI vector of shape (1, k_auto) where k_auto is the
                      number of latent semantic components. The vector is L2-normalized
                      to unit length.
        
        Process:
            1. Converts query to TF-IDF vector using the fitted vectorizer
            2. Projects the TF-IDF vector into LSI space using the fitted SVD model
            3. Reduces dimensionality by selecting only the first k_auto components
            4. Normalizes the resulting vector to unit length
        """
        query_tfidf = self.vectorizer.transform([query]) #transform query to tf-idf vector
        quert_lsi_full = self.svd_model.transform(query_tfidf) #project query tf-idf vector to LSI space
        query_lsi = quert_lsi_full[:, :self.k_auto] #select only the first k_auto components
        query_lsi = self.normaliser.transform(query_lsi) #normalize the query vector to unit length
        return query_lsi
    
    def cosine_sim(self, query_vec: np.ndarray)-> np.ndarray:
        def cosine_sim(self, query_vec: np.ndarray) -> np.ndarray:
            """
            Calculate the cosine similarity between document vectors and a query vector.
            
            Computes the dot product between all stored document vectors and the query vector,
            which represents the cosine similarity in the LSI (Latent Semantic Indexing) space.
            
            Args:
                query_vec (np.ndarray): A query vector of shape (n_features,) or (n_features, 1)
                                        representing the query in the LSI space.
            
            Returns:
                np.ndarray: A 1D array of similarity scores, one for each document in the collection.
                            Higher values indicate greater similarity to the query.
            """
        return (self.doc_vectors @ query_vec.T).flatten() #dot product between document vectors and query vector
     

    def save_models(self, folder: str = "models/lsi") -> None:
        """
        Save trained LSI models and related data to disk.

        This method persists the TF-IDF vectorizer, SVD model, normalizer, and other
        model components to a specified folder using joblib serialization. If document
        vectors and dataframe are available, they are also saved.

        Args:
            folder (str): Path to the folder where model files will be saved.
                          Defaults to "models/lsi". The folder is created if it
                          does not exist.

        Returns:
            None

        Raises:
            OSError: If the folder cannot be created or files cannot be written.
            Exception: If joblib serialization fails or CSV write fails.

        Notes:
            - Creates the following files in the specified folder:
                - tfidf_vectorizer.joblib: TF-IDF vectorizer object
                - svd_model.joblib: SVD/LSI model object
                - normaliser.joblib: Normalizer object
                - k_auto.joblib: Automatic k value
                - doc_vectors.npy: Document vectors (if available)
                - documents.csv: Document dataframe (if available)
            - Existing files with the same names will be overwritten.
        """
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
        """
        Load pre-trained LSI model components and document data from disk.
        
        This method loads all necessary artifacts for the LSI (Latent Semantic Indexing)
        model including the TF-IDF vectorizer, SVD model, normalizer, document vectors,
        and the document dataset.
        
        Args:
            folder (str, optional): Path to the folder containing the model files.
                Defaults to "models/lsi". Expected to contain the following files:
                - tfidf_vectorizer.joblib: Fitted TF-IDF vectorizer
                - svd_model.joblib: Fitted SVD model for dimensionality reduction
                - normaliser.joblib: Fitted normalizer for vector normalization
                - k_auto.joblib: Optimal number of components
                - doc_vectors.npy: Pre-computed document vectors
                - documents.csv: Original document dataset
        
        Returns:
            None
        
        Raises:
            FileNotFoundError: If any of the required model files are not found in the specified folder.
            joblib.load exceptions: If joblib cannot deserialize the model files.
            pd.errors.ParserError: If the CSV file is corrupted or cannot be parsed.
        """
        self.vectorizer = joblib.load(os.path.join(folder, "tfidf_vectorizer.joblib"))
        self.svd_model = joblib.load(os.path.join(folder, "svd_model.joblib"))
        self.normaliser = joblib.load(os.path.join(folder, "normaliser.joblib"))
        self.k_auto = joblib.load(os.path.join(folder, "k_auto.joblib"))

        self.doc_vectors = joblib.load(os.path.join(folder, "doc_vectors.npy"))
        self.df = pd.read_csv(os.path.join(folder, "documents.csv"))

       

