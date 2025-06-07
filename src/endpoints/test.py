from fastapi import APIRouter, HTTPException, status

from src.models.base.llm import load_llm, get_hf_llm

route = APIRouter(
    prefix='/test',
    tags=["Test API"]
)

@route.post('/load_model')
async def test_load_model():
    try:
        model_name = "gemini-2.0-flash"
        llm = load_llm(model_name)
        
        return {"status": "success", "message": f"Load model successfully: {model_name}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not supported: {str(e)}"
        )

@route.post('/load_hf_model')
async def test_load_hf_model():
    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        llm = get_hf_llm(model_name)

        return {"status": "success", "message": f"Load model successfully: {model_name}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not supported: {str(e)}"
        )