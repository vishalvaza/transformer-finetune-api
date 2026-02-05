from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_dir: str = "artifacts/model"  # where fine-tuned model is saved
    base_model: str = "distilbert-base-uncased"  # used only for cold-start dev
    device: str = "cpu"
    log_level: str = "INFO"

settings = Settings()
