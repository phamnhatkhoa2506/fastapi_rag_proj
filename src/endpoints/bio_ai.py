from fastapi import APIRouter, HTTPException, status

from src.rag.chain import OutputQA, InputQA, bio_chain

route = APIRouter(
    prefix='/bio-ai',
    tags=["Biology AI"]
)

@route.post("/invoke", response_model=OutputQA)
async def biology_ai(inputs: InputQA):
    try:
        answer = bio_chain.invoke(inputs.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )