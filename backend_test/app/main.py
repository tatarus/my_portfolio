from fastapi import FastAPI
from app.database import Base, engine
from app.users import router as users_router
from app.buisness import router as buisness_router

Base.metadata.create_all(bind=engine)  # создаёт таблицы

app = FastAPI()
app.include_router(users_router)
app.include_router(buisness_router)
