import pandas as pd
from nltk.corpus import stopwords
from greek_stemmer import stemmer 


class LoaderPreprocessor:
   

    greek_stopwords = set(stopwords.words('greek'))
    english_stopwords = set(stopwords.words('english'))
    all_stopwords = greek_stopwords.union(english_stopwords)

    """A class to load and preprocess data from a CSV file."""
    def __init__(self, file_path: str, pickSubset: bool = False, subsetPercent: float = 0.1, min_words: int = 30):
        """
        Initialize the LoaderPreprocessor instance.
        Args:
            file_path (str): The path to the data file to be loaded.
            pickSubset (bool, optional): Whether to use a subset of the data. Defaults to False.
            subsetPercent (float, optional): The percentage of data to use when pickSubset is True. 
                                             Defaults to 0.1 (10%).
        Attributes:
            file_path (str): The path to the data file.
            pickSubset (bool): Flag indicating whether a subset is being used.
            subsetPercent (float): The percentage of data to use for the subset.
            dataframe (pd.DataFrame): The loaded dataframe. Initialized as None.
            cleaned_dataframe (pd.DataFrame): The cleaned/processed dataframe. Initialized as None.
        """
        
        self.file_path = file_path
        self.pickSubset = pickSubset
        self.subsetPercent = subsetPercent
        self.min_words = min_words

        self.subsetPercent = self.__defineSubPercent()
        self.dataframe = None
        self.cleaned_dataframe = None
        


    


    def __defineSubPercent(self) -> float:
        """
        Determine the subset percentage for data loading.
        Validates and returns the appropriate subset percentage based on configuration.
        If subsetting is enabled, validates that subsetPercent is within the valid range [0.0, 1.0].
        If validation fails, returns a default value of 0.1.
        Returns:
            float: The subset percentage as a decimal between 0.0 and 1.0.
                Returns the configured subsetPercent if pickSubset is True and valid,
                returns 0.1 if pickSubset is True but subsetPercent is invalid,
                returns 1.0 if pickSubset is False (no subsetting).
        """
       
              
        if self.pickSubset:
            if self.subsetPercent < 0.0 or self.subsetPercent > 1.0:
                print("Error: subsetPercent must be between 0.0 and 1.0. Using default value of 0.1.")
                return 0.1
            return self.subsetPercent
        else:
            return 1.0

    

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load raw data and perform cleaning operations.
        
        This method loads raw data, creates a copy for cleaning, removes unwanted text characters,
        and drops specified columns related to parliamentary information. The original raw data
        and cleaned data are stored as instance attributes for later reference.
        
        Returns:
            pd.DataFrame: A cleaned DataFrame with text processed and specified columns removed.
        """

        df = self.load_data()

        df = self.drop_columns(df,columns=["parliamentary_sitting", "parliamentary_session"])

        df["speech"] = df["speech"].astype(str)
        df["raw_len"] = df["speech"].str.split().str.len()

        before = len(df)
        df = df[df["raw_len"]>=self.min_words]
        after = len(df)
        print(f"Dropped {before - after} rows with less than {self.min_words} words.")

        df.drop(columns=["raw_len"], inplace = True)
        

        df.rename(columns={"speech":"speech_raw"}, inplace=True)

        df["speech"] = df["speech_raw"].map(self.clean_text_string)
        self.cleaned_dataframe = df
        print("Data loaded and cleaned successfully.")
        return df

    #MAYBE CHANGE TO LOG FILES
    def load_data(self) -> pd.DataFrame:
        """
        Load data from a CSV file and optionally return a random subset.
        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded CSV data. If pickSubset
                        is True, returns a random sample of the data determined by subsetPercent.
                        Returns an empty DataFrame if an error occurs during loading.
        Raises:
            Prints error message to console if file reading fails, but does not raise an exception.
        Notes:
            - Uses UTF-8 encoding for reading the CSV file
            - If pickSubset is True, uses random_state=42 for reproducible sampling
            - Index is reset on returned DataFrame(s)
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
        """
        Clean and preprocess text data in the 'speech' column of a DataFrame.
        This method performs the following transformations on the 'speech' column:
        1. Converts all text to lowercase
        2. Removes all non-word characters (punctuation, special characters)
        3. Removes common stopwords from the text
        Args:
            df (pd.DataFrame): Input DataFrame containing a 'speech' column with text data.
        Returns:
            pd.DataFrame: DataFrame with the cleaned 'speech' column. The original DataFrame
                          is modified in-place and returned.
        Example:
            >>> df = pd.DataFrame({'speech': ['Hello, World!', 'Python Programming!']})
            >>> cleaned_df = self.__clean_text(df)
            >>> print(cleaned_df['speech'])
        """
        
        speech = df['speech'].astype(str).str.lower()
        speech = speech.str.replace(r'[^\w\s]', '', regex=True)
        speech = speech.apply(self.__remove_stopwords)
        df['speech'] = speech
        return df
    
    def __remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the input text and apply stemming to the remaining words.
        This method filters out common stopwords from the provided text and then
        applies stemming to each of the remaining words to reduce them to their
        root form.
        Args:
            text (str): The input text from which stopwords should be removed.
        Returns:
            str: A string containing the stemmed words with stopwords removed,
                 joined by spaces.
        Example:
            >>> result = self.__remove_stopwords("The quick brown fox jumps")
            >>> # Returns stemmed words without common stopwords like "the"
        """
        
   
        words = text.split()
        filtered_words = [word for word in words if word not in self.all_stopwords]
        #stemmed_words = [self.__stem_text(word) for word in filtered_words]
        
        return ' '.join(filtered_words)

    def __stem_text(self, word: str) -> str:
        """
        Stem a word if it is in Greek language, otherwise return the word as-is.
        For Greek words, applies stemming with the VBG (verb gerund) tag and converts
        the result to lowercase. For non-Greek words, returns the original word unchanged.
        Args:
            word (str): The word to be stemmed.
        Returns:
            str: The stemmed word in lowercase if Greek, otherwise the original word.
        """
        
        #print(f"Stemming word: {word}")
        if self.is_greek(word):
            return stemmer.stem_word(word, 'VBG').lower()
        return word

    def is_greek(self, word: str) -> bool:
        """
        Check if a word contains any Greek characters.
        
        Determines whether the given word contains at least one character
        from the Greek Unicode ranges (Basic Greek: U+0370-U+03FF or
        Greek Extended: U+1F00-U+1FFF).
        
        Args:
            word (str): The word to check for Greek characters.
        
        Returns:
            bool: True if the word contains at least one Greek character,
                  False otherwise.
        
        Examples:
            >>> is_greek("Ελληνικά")
            True
            >>> is_greek("hello")
            False
            >>> is_greek("café")
            False
        """
        for char in word:
            if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF':
                return True
        return False
    
    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drop specified columns from a DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame from which columns will be dropped.
            columns (list): A list of column names to drop from the DataFrame.
        Returns:
            pd.DataFrame: A new DataFrame with the specified columns removed.
                        If a column in the list does not exist, it is ignored.
        Notes:
            - This method uses errors='ignore' to suppress errors when attempting
            to drop columns that don't exist in the DataFrame.
            - The original DataFrame is not modified; a new DataFrame is returned.
        """
     
        df = df.drop(columns=columns, errors='ignore')
        return df

    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """
        Save a cleaned pandas DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): The cleaned DataFrame to save.
            output_path (str): The file path where the CSV file will be saved.
        
        Returns:
            None
        
        Raises:
            Exception: Prints error message if the file cannot be saved.
        
        Example:
            >>> cleaned_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            >>> loader.save_cleaned_data(cleaned_df, 'output/cleaned_data.csv')
            Cleaned data saved to output/cleaned_data.csv
        """
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Cleaned data saved to {output_path}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")

    def clean_text_string(self, text: str) -> str:
        """
        Clean and preprocess a text string by normalizing it for text processing.
        
        Converts the input to lowercase, removes non-alphanumeric characters (except spaces),
        and filters out common stopwords.
        
        Args:
            text (str): The input text to clean. If not a string, it will be converted to one.
        
        Returns:
            str: The cleaned text string with lowercase letters, numbers, spaces only,
                 and stopwords removed.
        """

        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        text = self.__remove_stopwords(text)
        return text

