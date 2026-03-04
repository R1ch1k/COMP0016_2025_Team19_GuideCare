import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.core.config import settings
from app.db import models

DATABASE_URL = settings.DATABASE_URL

async_engine = create_async_engine(DATABASE_URL, echo=False, future=True)

AsyncSessionLocal = sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
)

async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    # Create tables
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
