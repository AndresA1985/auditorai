import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("DB_HOST", "127.0.0.1")
    db_port: int = int(os.getenv("DB_PORT", "3307"))
    db_database: str = os.getenv("DB_DATABASE", "sis_medico")
    db_username: str = os.getenv("DB_USERNAME", "sistema_medico")
    db_password: str = os.getenv("DB_PASSWORD", "")
    min_score: float = float(os.getenv("AUDITORIA_MIN_SCORE", "0.08"))


settings = Settings()