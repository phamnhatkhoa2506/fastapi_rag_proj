from fastapi import APIRouter, HTTPException, status

from src.models.base.llm import load_llm

route = APIRouter(
    prefix='/test',
    tags=["Test API"]
)

@route.post('/load_model')
def test_load_model():
    try:
        llm = load_llm("gemini-2.0-flash")

        return HTTPException(
            status_code=status.HTTP_200_OK,
            detail="Load model successfully"
        )
    except:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not support"
        )