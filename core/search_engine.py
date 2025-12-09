import pandas as pd
from core.loader_preprocessor import LoaderPreprocessor
from core.vector_lsi_model import VectorLSIModel
import numpy as np

class Search_Engine:
    def __init__(self, model_dir: str = "models/lsi_model/", text_col: str = "speech", target: float = 0.72):
        
        self.lsi = VectorLSIModel(text_col=text_col, target=target)
        self.lsi.load_models(model_dir)

        self.preprocessor = LoaderPreprocessor(file_path=None)




    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        cleaned_query = self.preprocessor.clean_text_string(query)
        query_vec = self.lsi.query_vector(cleaned_query)
        sims = self.lsi.cosine_sim(query_vec)
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.lsi.df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        return results
    

    ## MAYBE NEED TO ADD MORE METHODS FOR RETURNING FILTERED SEARCH RESULTS BASED ON PARTY, DATE, NAME, ETC.