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


def setup_logging() -> None:
    """Configure Loguru structured logging."""
    logger.remove()

    # Formatted console sink
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
    )

    # Structured JSON log file sink
    logger.add(
        "logs/aiforge_{time:YYYY-MM-DD}.log",
        rotation="50 MB",
        retention="10 days",
        compression="zip",
        serialize=True,
        level=settings.LOG_LEVEL,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for uvicorn_logger in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logging.getLogger(uvicorn_logger).handlers = [InterceptHandler()]
