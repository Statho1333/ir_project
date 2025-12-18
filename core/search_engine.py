import pandas as pd
from core.loader_preprocessor import LoaderPreprocessor
from core.vector_lsi_model import VectorLSIModel
import numpy as np

class Search_Engine:
    def __init__(self, model_dir: str = "models/lsi_model/", text_col: str = "speech", target: float = 0.72):
        """
        Initialize the search engine with LSI model and text preprocessor.
        Args:
            model_dir (str, optional): Directory path containing the LSI model files. 
                Defaults to "models/lsi_model/".
            text_col (str, optional): Name of the text column to use for vectorization. 
                Defaults to "speech".
            target (float, optional): Target value for LSI model configuration. 
                Defaults to 0.72.
        Attributes:
            lsi (VectorLSIModel): Latent Semantic Indexing model instance.
            preprocessor (LoaderPreprocessor): Text preprocessor instance.
        """
        
        self.lsi = VectorLSIModel(text_col=text_col, target=target)
        self.lsi.load_models(model_dir)

        self.preprocessor = LoaderPreprocessor(file_path=None)




    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Search for documents similar to the given query using LSI (Latent Semantic Indexing).
        
        Args:
            query (str): The search query string to process and match against the document corpus.
            top_k (int, optional): The maximum number of top results to return. Defaults to 10.
        
        Returns:
            pd.DataFrame: A DataFrame containing the top_k most similar documents with their 
                         corresponding similarity scores in a 'score' column, sorted by relevance 
                         in descending order.
        
        Process:
            1. Cleans the input query text using the preprocessor
            2. Converts the cleaned query to a vector representation using LSI
            3. Computes cosine similarity between the query vector and all documents
            4. Retrieves the top_k most similar documents
            5. Adds similarity scores to the results DataFrame
        """
        cleaned_query = self.preprocessor.clean_text_string(query)
        query_vec = self.lsi.query_vector(cleaned_query)
        sims = self.lsi.cosine_sim(query_vec)
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.lsi.df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        return results
    

    ## MAYBE NEED TO ADD MORE METHODS FOR RETURNING FILTERED SEARCH RESULTS BASED ON PARTY, DATE, NAME, ETC.