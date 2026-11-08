"""Entrypoint: configure logging FIRST, then build the app. Run: `python -m app.main`
(or `uvicorn app.main:app`). Uvicorn target = `app.main:app`."""

from app.config import get_settings
from app.logging_setup import configure_logging

_settings = get_settings()
configure_logging(_settings.LOG_LEVEL)

from app.registrar import register_app  # noqa: E402  (must come after logging is configured)

app = register_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_settings.PORT)
