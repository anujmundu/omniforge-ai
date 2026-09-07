from apps.api.routers.health import router as health_router
from apps.api.routers.auth import router as auth_router
from apps.api.routers.projects import router as projects_router
from apps.api.routers.datasets import router as datasets_router
from apps.api.routers.experiments import router as experiments_router
from apps.api.routers.ml import router as ml_router
from apps.api.routers.vision import router as vision_router
from apps.api.routers.nlp import router as nlp_router

__all__ = [
    "health_router",
    "auth_router",
    "projects_router",
    "datasets_router",
    "experiments_router",
    "ml_router",
    "vision_router",
    "nlp_router",
]
