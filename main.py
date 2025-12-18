from typing import Union
from core.search_engine import Search_Engine
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

def main():
    engine = Search_Engine()
    print("Search Engine is ready to use.")

    results = engine.search("φορολογία επιχειρήσεων", top_k=5)
    for r in results:
        print(r)

if __name__ == "__main__":
    main()