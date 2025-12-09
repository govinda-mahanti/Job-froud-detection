# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-prod')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True