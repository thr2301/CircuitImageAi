from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from services.ml_manager import ml_manager
from routers.auth import router as auth_router
from routers.ml_routes import router as ml_router
from routers.engineering import router as engineering_router, load_vowel_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vowel_data()
    ml_manager.load_model()
    yield
    ml_manager.unload_model()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(auth_router)
app.include_router(ml_router)
app.include_router(engineering_router)