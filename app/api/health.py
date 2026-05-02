from fastapi import APIRouter, status
from fastapi.responses import Response

router = APIRouter(prefix="", tags=["health"])


@router.post("/ready", status_code=status.HTTP_200_OK)
def read_root():
    return Response(status_code=status.HTTP_200_OK)
