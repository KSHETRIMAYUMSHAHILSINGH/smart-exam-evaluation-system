from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "sqlite:///./data/exam_eval.db"
    api_key: str = "dev-local-key-change-me"
    max_score: int = 10

settings = Settings()