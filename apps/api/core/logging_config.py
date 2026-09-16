import logging
import sys

from loguru import logger

from apps.api.core.config import get_settings

settings = get_settings()


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and redirect them to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


from typing import Optional


def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """Configure Loguru structured logging."""
    logger.remove()
    effective_level = log_level or settings.LOG_LEVEL
    effective_file = log_file or getattr(settings, "LOG_FILE", "logs/aiforge_{time:YYYY-MM-DD}.log")

    # Formatted console sink
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=effective_level,
    )

    # Structured JSON log file sink
    logger.add(
        effective_file,
        rotation="50 MB",
        retention="10 days",
        compression="zip",
        serialize=True,
        level=effective_level,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for uvicorn_logger in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logging.getLogger(uvicorn_logger).handlers = [InterceptHandler()]
