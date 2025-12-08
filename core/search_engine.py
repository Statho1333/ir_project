from core.loader_preprocessor import Loader_Preprocessor
from core.vector_lsi_model import Vector_LSI_Model
import pandas as pd


class Search_Engine:
    def __init__(self, csv_path: str, query: str, name: str = "SearchEngine"):
        self.name = name
        self.csv_path = csv_path
        self.query = query


    def __build_engine(self) -> None:
        # Load and preprocess data
        loader = Loader_Preprocessor(self.csv_path, pickSubset=False)
        df = loader.cleaned_dataframe

        # Build LSI model
        lsi_model = Vector_LSI_Model(text_col='cleaned_text', target=0.72)
        lsi_model.fit(df) #fit the model to the dataframe
        lsi_model.fit_lsi() #perform SVD and reduce dimensionality

        self.lsi_model = lsi_model #store the trained LSI model
      
    def search(self, query):
        # Placeholder for search implementation
        pass

    #TODO : NEED TO IMPLEMENET IT, DONT CONSIDER IT YET