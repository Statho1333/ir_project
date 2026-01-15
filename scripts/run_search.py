import pandas as pd
from core.search_engine import Search_Engine

class RunSearchService:
    def __init__(self):
        """Initialize the search service by loading the search engine and data."""
        
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
        """Return a list of unique political parties."""

        if "political_party" in self.df.columns:
            return sorted(self.df["political_party"].dropna().astype(str).str.upper().unique().tolist())
        return []

    def get_mps(self):
        """Return a list of unique member names."""

        if "member_name" in self.df.columns:
            return sorted(self.df["member_name"].dropna().astype(str).str.upper().unique().tolist())
        return []

    def get_year_range(self):
        """Return the min and max year from the sitting_date column."""

        years = self.df["sitting_date"].dropna().dt.year
        
        if years.empty:
            return 0,3000
            
        return int(years.min()), int(years.max())
    