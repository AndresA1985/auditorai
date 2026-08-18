import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def required_env(name: str, *, allow_empty: bool = False) -> str:
    value = os.getenv(name)
    if value is None or (value == "" and not allow_empty):
        raise RuntimeError("Falta configurar {0} en el archivo .env".format(name))
    return value


def env_int(name: str) -> int:
    value = required_env(name)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError("{0} debe ser un numero entero en .env".format(name)) from exc


def env_float(name: str) -> float:
    value = required_env(name)
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError("{0} debe ser un numero decimal en .env".format(name)) from exc


def env_path(name: str) -> Path:
    value = Path(required_env(name))
    if value.is_absolute():
        return value
    return ROOT_DIR / value


def env_str(name: str) -> str:
    return required_env(name).strip()


@dataclass(frozen=True)
class Settings:
    db_host: str = required_env("DB_HOST")
    db_port: int = env_int("DB_PORT")
    db_database: str = required_env("DB_DATABASE")
    db_username: str = required_env("DB_USERNAME")
    db_password: str = required_env("DB_PASSWORD", allow_empty=True)
    min_score: float = env_float("AUDITORIA_MIN_SCORE")
    model_path: Path = env_path("AUDITORIA_MODEL_PATH")
    model_min_similarity: float = env_float("AUDITORIA_MODEL_MIN_SIMILARITY")
    score_exact_match_bonus: float = env_float("AUDITORIA_SCORE_EXACT_MATCH_BONUS")
    score_token_match_bonus: float = env_float("AUDITORIA_SCORE_TOKEN_MATCH_BONUS")
    score_code_group_bonus: float = env_float("AUDITORIA_SCORE_CODE_GROUP_BONUS")
    validation_size: float = env_float("AUDITORIA_VALIDATION_SIZE")
    test_size: float = env_float("AUDITORIA_TEST_SIZE")
    model_engine: str = env_str("AUDITORIA_MODEL_ENGINE")
    embedding_model_name: str = env_str("AUDITORIA_EMBEDDING_MODEL")
    embedding_device: str = env_str("AUDITORIA_EMBEDDING_DEVICE")
    embedding_batch_size: int = env_int("AUDITORIA_EMBEDDING_BATCH_SIZE")


settings = Settings()
