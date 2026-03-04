import sys
import os
import asyncio

# Ensure the app package (backend/src) is importable when run from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app.db.session import init_db

if __name__ == "__main__":
    """Run this script to (re)create database tables using the app's metadata.

    Usage:
      # Use an env var to point to a test DB if desired:
      TEST_DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/guidecare_test" python backend/scripts/init_db.py

    If TEST_DATABASE_URL is provided it will be used to override DATABASE_URL for this run.
    The application otherwise picks up DATABASE_URL (or the default in app.core.config.Settings).
    """
    # Allow quick overrides for test runs
    test_db = os.environ.get("TEST_DATABASE_URL")
    if test_db:
        os.environ["DATABASE_URL"] = test_db
    asyncio.run(init_db())
