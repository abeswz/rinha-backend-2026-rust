from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    app_name: str = "fraud-detection-rinha-backend-2026"
    debug: bool = False


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
