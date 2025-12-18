import pandas as pd
from core.loader_preprocessor import LoaderPreprocessor
from core.vector_lsi_model import VectorLSIModel
import numpy as np

class Search_Engine:
    def __init__(self, model_dir: str = "models/lsi_model/", text_col: str = "speech", target: float = 0.72):
        """
        Initialize the search engine with LSI model and preprocessor.
        Args:
            model_dir (str, optional): Directory path where LSI models are stored. 
                Defaults to "models/lsi_model/".
            text_col (str, optional): Name of the text column to use for LSI model. 
                Defaults to "speech".
            target (float, optional): Target parameter for the LSI model. 
                Defaults to 0.72.
        """
 
        
        self.lsi = VectorLSIModel(text_col=text_col, target=target)
        self.lsi.load_models(model_dir)

        self.preprocessor = LoaderPreprocessor(file_path=None)




    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Search for documents similar to the given query using LSI (Latent Semantic Indexing).
        Args:
            query (str): The search query string.
            top_k (int, optional): The number of top results to return. Defaults to 10.
        Returns:
            pd.DataFrame: A DataFrame containing the top_k most similar documents with their similarity scores.
                          Includes all original columns from the index plus a "score" column with cosine similarity values.
        Raises:
            None
        Example:
            >>> results = search_engine.search("machine learning", top_k=5)
            >>> print(results[["title", "score"]])
        """
        
        cleaned_query = self.preprocessor.clean_text_string(query)
        query_vec = self.lsi.query_vector(cleaned_query)
        sims = self.lsi.cosine_sim(query_vec)
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.lsi.df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        return results
    

    ## MAYBE NEED TO ADD MORE METHODS FOR RETURNING FILTERED SEARCH RESULTS BASED ON PARTY, DATE, NAME, ETC.