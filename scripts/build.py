import pandas as pd
from core.loader_preprocessor import LoaderPreprocessor
from core.vector_lsi_model import VectorLSIModel   

if __name__ == "__main__":
    
    raw_data_path = "data/raw/Greek_Parliament_Proceedings_1989_2019.csv"
    cleaned_data_path = "data/processed/cleaned_data.csv"
    model_dir = "models/lsi_model/"

    ##Load and preprocess data
    lp = LoaderPreprocessor(file_path=raw_data_path, pickSubset=False)
    cleaned_df = lp.load_and_clean()

    # Save cleaned data
    lp.save_cleaned_data(cleaned_df,cleaned_data_path)

    #lsi model
    lsi_model = VectorLSIModel(text_col="speech", target=0.72)
    lsi_model.fit(cleaned_df) #fit the model to the dataframe
    lsi_model.fit_lsi(k_cap=100) #perform SVD and reduce dimensionality

    lsi_model.save_models(model_dir)