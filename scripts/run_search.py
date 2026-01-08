import pandas as pd
from core.search_engine import Search_Engine

class RunSearchService:
    def __init__(self):
        self.engine = Search_Engine()
        self.df = self.engine.lsi.df

        # Normalize date's form
        self.df["sitting_date"] = pd.to_datetime(
            self.df["sitting_date"],
            format="%Y-%m-%d",
            errors="coerce"
        )

    # ---------------------------
    # FILTERS
    # ---------------------------

    def get_parties(self):
        if "political_party" in self.df.columns:
            return sorted(self.df["political_party"].dropna().astype(str).str.upper().unique().tolist())
        return []

    def get_mps(self):
        if "member_name" in self.df.columns:
            return sorted(self.df["member_name"].dropna().astype(str).str.upper().unique().tolist())
        return []

    def get_year_range(self):
        years = self.df["sitting_date"].dropna().dt.year
        
        if years.empty:
            return 0,3000
            
        return int(years.min()), int(years.max())
    