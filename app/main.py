from fastapi import FastAPI, status
from fastapi.responses import Response

app = FastAPI()


# Just return ready
@app.get("/ready", status_code=status.HTTP_200_OK)
def read_root():
    return Response(status_code=status.HTTP_200_OK)
