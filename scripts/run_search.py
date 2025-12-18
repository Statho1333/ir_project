from core.search_engine import Search_Engine

def main():
    engine = Search_Engine()
    print("Search Engine is ready to use.")

def __init__(self):
        self.engine = Search_Engine()
        self.df = self.engine.lsi.df  

# ---------------------------
    # FILTERxS
    # ---------------------------

def get_parties(self):
        return sorted(self.df["party"].dropna().unique().tolist())

def get_mps(self):
        return sorted(self.df["mp_name"].dropna().unique().tolist())

def get_year_range(self):
        years = self.df["date"].dropna().astype(str).str[:4].astype(int)
        return years.min(), years.max()