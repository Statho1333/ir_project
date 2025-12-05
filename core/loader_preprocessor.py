import pandas as pd
from ntlk.corpus import stopwords
from greek_stemmer import GreekStemmer


class LoaderPreprocessor:

    greek_stopwords = set(stopwords.words('greek'))
    english_stopwords = set(stopwords.words('english'))
    all_stopwords = greek_stopwords.union(english_stopwords)

    """A class to load and preprocess data from a CSV file."""
    def __init__(self, file_path: str, pickSubset: bool = False, subsetPercent: float = 0.1):
        self.stemmer = GreekStemmer()
        self.file_path = file_path
        self.pickSubset = pickSubset
        self.subsetPercent = subsetPercent

        self.subsetPercent = self.__defineSubPercent()
        self.dataframe = self.load_data()
        self.cleaned_dataframe = self.__clean_text(self.dataframe)


    


    def __defineSubPercent(self) -> float:
        """
        Determines the percentage of the dataset to use based on the `pickSubset` flag and `subsetPercent` value.
        Returns:
            float: The percentage of the dataset to use. If `pickSubset` is True and `subsetPercent` is between 0.0 and 1.0 (inclusive),
            returns `subsetPercent`. If `subsetPercent` is out of bounds, prints an error and returns 0.1. If `pickSubset` is False,
            returns 1.0 (use the entire dataset).
        """
              
        if self.pickSubset:
            if self.subsetPercent < 0.0 or self.subsetPercent > 1.0:
                print("Error: subsetPercent must be between 0.0 and 1.0. Using default value of 0.1.")
                return 0.1
            return self.subsetPercent
        else:
            return 1.0

    
    #MAYBE CHANGE TO LOG FILES
    def load_data(self) -> pd.DataFrame:
        """
        Load data from a CSV file with optional subset sampling.
        
        Reads a CSV file from the specified filepath with UTF-8 encoding.
        If subset sampling is enabled, returns a random sample of the data
        based on the configured subset percentage. Otherwise, returns the
        complete dataset.
        
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data. If a subset
                is requested, returns a sampled DataFrame. Returns an empty
                DataFrame if an error occurs during loading.
        
        Raises:
            Prints error message to console if file loading fails, but does
            not raise an exception.
        """
        try:
            df = pd.read_csv(self.file_path, encoding='utf-8')
            if self.pickSubset:
                df_subset = df.sample(n=int(len(df) * self.subsetPercent), random_state=42)
                return df_subset.reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
        
    def __clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        
        speech = df['speech'].astype(str).str.lower()
        speech = speech.str.replace(r'[^\w\s]', '', regex=True)
        speech = speech.apply(self.__remove_stopwords)
        df['speech'] = speech
        return df
    
    def __remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the given text.
        The method splits the input string on whitespace, filters out any token that is present
        in self.all_stopwords, and returns the remaining tokens joined with a single space.
        Args:
            text (str): The input text to process. Tokenization is performed using str.split()
                        (i.e., split on any whitespace).
        Returns:
            str: The input text with stopwords removed. If all tokens are removed, an empty string
                is returned.
        Notes:
            - Matching against self.all_stopwords is exact and therefore sensitive to casing and
            punctuation. Normalize text (e.g., lowercase, strip punctuation) beforehand if
            desired.
            - This method does not raise exceptions for empty or non-string input; callers should
            ensure a string is provided.
        """
   
        words = text.split()
        filtered_words = [word for word in words if word not in self.all_stopwords]
        stemmed_words = [self.__stem_text(word) for word in filtered_words]
        
        return ' '.join(stemmed_words)

    def __stem_text(self, word: str) -> str:
        if self.is_greek(word):
            return self.stemmer.stem(word)
        return word

    def is_greek(self, word: str) -> bool:
        for char in word:
            if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF':
                return True
        return False
    
    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
     
        df = df.drop(columns=columns, errors='ignore')
        return df

    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Cleaned data saved to {output_path}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")
