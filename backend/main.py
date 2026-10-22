from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException

from data import Data
from llm import LLM

data_service = Data()
llm_service = LLM()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load course embeddings when the server starts."""
    data_service.load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """Check that the app is running."""
    return {"message": "app is running"}


def retrieved(query: str, top_k: int):
    """Search the course index, or report that it is not loaded."""
    try:
        return data_service.search(query, topk=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/v1/search")
async def search(
    query: str,
    top_k: int = 5,
    mode: Literal["dense"] = "dense",
):
    """Search courses with dense retrieval."""
    hits = retrieved(query, top_k)
    return {
        "query": query,
        "mode": mode,
        "results": [
            {
                "course_code": course.course_code,
                "title": course.title,
                "score": score,
            }
            for course, score in hits
        ],
    }


@app.get("/v1/ask")
async def ask(
    query: str,
    top_k: int = 5,
    mode: Literal["dense"] = "dense",
):
    """Answer a question from the closest courses."""
    hits = retrieved(query, top_k)
    try:
        answer = llm_service.query(query, hits)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "query": query,
        "answer": answer,
        "retrieved_courses": [
            {"course_code": course.course_code, "title": course.title}
            for course, _score in hits
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
