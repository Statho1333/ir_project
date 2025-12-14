from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from collections import Counter
import re

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# -------------------------
# Helper functions (mock)
# -------------------------
def simple_text_search(q: str):
    """
    Mock search results for GUI purposes
    """
    sample = []
    for i in range(1,6):
        sample.append({
            "speech_id": f"mock-{i}",
            "mp_name": f"Βουλευτής {i}",
            "party": "Δείγμα",
            "date": "2010-01-01",
            "text": f"Αυτό είναι ένα υποθετικό απόσπασμα με την λέξη {q} και περισσότερο κείμενο."
        })
    return sample

def extract_top_terms(text: str, topn: int = 10):
    toks = re.findall(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋ\w']+", text.lower())
    toks = [t for t in toks if len(t) > 2]
    stop = set(["κύριε","κυρία","κ.","και","του","της","το","των","στο","στη","στον","παρουσία","σας"])
    toks = [t for t in toks if t not in stop]
    ctr = Counter(toks)
    return ctr.most_common(topn)

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Form(...), party: str = Form(None), date_from: str = Form(None), date_to: str = Form(None)):
    results = simple_text_search(q)
    combined_text = " ".join([r["text"] for r in results])
    top_terms = extract_top_terms(combined_text, topn=15)

    context = {
        "request": request,
        "query": q,
        "results": results,
        "count": len(results),
        "top_terms": top_terms,
        "party": party or "",
        "date_from": date_from or "",
        "date_to": date_to or "",
    }
    return templates.TemplateResponse("results.html", context)

@app.get("/speech/{speech_id}", response_class=HTMLResponse)
async def speech_detail(request: Request, speech_id: str):
    rec = {
        "speech_id": speech_id,
        "mp_name": "Δείγμα Βουλευτή",
        "party": "Δείγμα",
        "date": "2015-05-01",
        "text": f"Αυτό είναι ένα υποθετικό πλήρες κείμενο ομιλίας με id {speech_id}."
    }
    keywords = extract_top_terms(rec["text"], topn=20)
    return templates.TemplateResponse("speech.html", {"request": request, "rec": rec, "keywords": keywords})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
