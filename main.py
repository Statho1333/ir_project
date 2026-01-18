from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from collections import Counter
import re
import pandas as pd

from scripts.run_search import RunSearchService


# Create FastAPI application instance (entry point of the backend)
app = FastAPI()

# Initialize the search service once at startup
# This loads data and trained models into memory
print(">>> Initializing search service...")
service = RunSearchService()
print(">>> Ready")


# -------------------------
# Static files & templates
# -------------------------

# Serve static assets (CSS files)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Configure Jinja2 template directory
templates = Jinja2Templates(directory="frontend/templates")


# -------------------------
# Helper functions
# -------------------------

def extract_top_terms(text: str, topn: int = 10):
    """
    Extracts the most frequent terms from a given text.
    Used to display keywords in the results page
    and in the individual speech view.
    """
    if not text or not isinstance(text, str):
        return []

    # Tokenization: extract Greek and Latin words
    toks = re.findall(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋ\w']+", text.lower())

    # Remove very short tokens
    toks = [t for t in toks if len(t) > 2]

    # Simple stopword list
    stop = {
        "κύριε", "κυρία", "και", "του", "της", "το",
        "των", "στο", "στη", "στον", "παρουσία", "σας", "είναι", "από"
    }

    toks = [t for t in toks if t not in stop]

    # Return the top-N most frequent terms
    return Counter(toks).most_common(topn)


# -------------------------
# Routes
# -------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Home page route.
    Displays the search form and dynamically loads
    available parties, MPs and year range.
    """
    parties = service.get_parties()
    mps = service.get_mps()
    years = service.get_year_range()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "parties": parties,
            "mps": mps,
            "years": years
        }
    )


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = Form(...),
    party: str = Form(None),
    mp: str = Form(None),
    date_from: str = Form(None),
    date_to: str = Form(None),
):
    """
    Executes a search query and applies optional filters
    such as political party, MP and date range.
    """

    # Perform information retrieval using the search engine
    df = service.engine.search(q, top_k=30)

    # -------- Apply filters --------
    if party:
        df = df[df["political_party"].astype("string").str.strip().str.upper() == party]

    if mp:
        df = df[df["member_name"].astype("string").str.strip().str.upper() == mp]

    if date_from:
        date_from_dt = pd.to_datetime(date_from)
        df = df[df["sitting_date"] >= date_from_dt]

    if date_to:
        date_to_dt = pd.to_datetime(date_to)
        df = df[df["sitting_date"] <= date_to_dt]

    # Prepare results for front-end rendering
    results = []
    for idx, row in df.iterrows():
        raw_name = str(row.get("member_name", "Unknown"))
        raw_party = str(row.get("political_party", "---"))

        # Formatting for presentation
        formatted_name = raw_name.title()
        formatted_party = raw_party.upper()

        results.append({
            "speech_id": str(idx),
            "mp_name": formatted_name,
            "party": formatted_party,
            "speech": row.get("speech_raw", row.get("speech", "")),
            "date": row["sitting_date"].strftime('%d/%m/%Y')
                    if pd.notnull(row["sitting_date"]) else "---"
        })

    # Extract aggregate keywords from all retrieved speeches
    combined_text = " ".join(r["speech"] for r in results)
    top_terms = extract_top_terms(combined_text, 15)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": q,
            "results": results,
            "count": len(results),
            "top_terms": top_terms,
            "party": party or "",
            "mp": mp or "",
            "date_from": date_from or "",
            "date_to": date_to or ""
        }
    )


@app.get("/speech/{speech_id}", response_class=HTMLResponse)
async def speech_detail(request: Request, speech_id: str):
    """
    Displays the full text of a single speech
    identified by its unique ID.
    """
    try:
        # Retrieve speech record from the original dataset
        idx = int(speech_id)
        row = service.df.loc[idx]

        rec = {
            "mp_name": row.get("member_name", "Unknown"),
            "party": row.get("political_party", "---"),
            "date": row["sitting_date"].strftime('%d/%m/%Y')
                    if pd.notnull(row["sitting_date"]) else "---",
            "speech": row.get("speech_raw", "Speech text not found")
        }

        # Extract keywords for the specific speech
        keywords = extract_top_terms(rec["speech"], 20)

        return templates.TemplateResponse(
            "speech.html",
            {
                "request": request,
                "rec": rec,
                "keywords": keywords
            }
        )

    except Exception as e:
        # Error handling for invalid or missing speech IDs
        print(f"Error finding speech: {e}")
        return HTMLResponse(content="Speech not found", status_code=404)
