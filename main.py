from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from collections import Counter
import re
import pandas as pd

from scripts.run_search import RunSearchService

app = FastAPI()

print(">>> Initializing search service...")
service = RunSearchService()
print(">>> Ready")

# -------------------------
# Static files & templates
# -------------------------
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# -------------------------
# Helpers
# -------------------------
def extract_top_terms(text: str, topn: int = 10):
    if not text or not isinstance(text, str):
        return []
    toks = re.findall(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋ\w']+", text.lower())
    toks = [t for t in toks if len(t) > 2]

    stop = {
        "κύριε", "κυρία", "και", "του", "της", "το",
        "των", "στο", "στη", "στον", "παρουσία", "σας", "είναι", "από"
    }

    toks = [t for t in toks if t not in stop]
    return Counter(toks).most_common(topn)

# -------------------------
# Routes
# -------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
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
    # Searching in the engine
    df = service.engine.search(q, top_k=30)

    # -------- Setting filters --------
    if party:
        df = df[df["political_party"] == party]

    if mp:
        df = df[df["member_name"] == mp]

    if date_from:
        date_from_dt = pd.to_datetime(date_from)
        df = df[df["sitting_date"] >= date_from_dt]
    
    if date_to:
        date_to_dt = pd.to_datetime(date_to)
        df = df[df["sitting_date"] <= date_to_dt]

    # Creating a list with the results
    results = []
    for idx, row in df.iterrows():
        raw_name = str(row.get("member_name", "Άγνωστος"))
        raw_party = str(row.get("political_party", "---"))

        # Modify names to be dispalyed properly
        formatted_name = raw_name.title()
        formatted_party = raw_party.upper() 

        results.append({
            "speech_id": str(idx), 
            "mp_name": formatted_name,
            "party": formatted_party,
            "speech": row.get("speech", ""),
            "date": row["sitting_date"].strftime('%d/%m/%Y') if pd.notnull(row["sitting_date"]) else "---"
        })

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
    try:
        idx = int(speech_id)
        row = service.df.iloc[idx]
        rec = {
            "mp_name": row.get("member_name", "Άγνωστος"),
            "party": row.get("political_party", "---"),
            "date": row["sitting_date"].strftime('%d/%m/%Y') if pd.notnull(row["sitting_date"]) else "---",
            "speech": row.get("speech", "Δεν βρέθηκε κείμενο")
        }

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
        print(f"Error finding speech: {e}")
        return HTMLResponse(content="Η ομιλία δεν βρέθηκε", status_code=404)