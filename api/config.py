"""
Configuration validation module for OCR API.
Validates all environment variables and configuration at startup.
"""
import os
import sys
from pathlib import Path

class ConfigValidator:
    """Validates and provides access to configuration."""
    
    @staticmethod
    def validate_paths():
        """Validate that required directories exist or can be created."""
        required_dirs = {
            "LOGS_DIR": "/app/logs",
            "SCHEMA_DIR": "/app/schemas",
            "DATA_DIR": "/app/data",
        }
        
        for name, path in required_dirs.items():
            env_path = os.environ.get(name, path)
            try:
                os.makedirs(env_path, exist_ok=True)
            except Exception as e:
                print(f"WARNING: Could not create directory {name}={env_path}: {e}")
    
    @staticmethod
    def validate_credentials():
        """Check if required credentials are available."""
        issues = []
        
        # Check Firebase
        firebase_key = os.environ.get("FIREBASE_KEY_PATH", "firebase-key.json")
        if not os.path.exists(firebase_key):
            issues.append(f"Firebase key not found: {firebase_key}")
        
        # Check Google Vision
        google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if google_creds and not os.path.exists(google_creds):
            issues.append(f"Google Vision credentials not found: {google_creds}")
        
        return issues
    
    @staticmethod
    def validate_environment():
        """Validate environment variables."""
        config = {
            "PADDLE_OCR_URL": os.environ.get("PADDLE_OCR_URL", "http://paddle-ocr-service:8001/ocr"),
            "GOOGLE_VISION_URL": os.environ.get("GOOGLE_VISION_URL", "http://google-vision-service:8003/ocr"),
            "TESSERACT_OCR_URL": os.environ.get("TESSERACT_OCR_URL", "http://tesseract-ocr-service:8002/ocr"),
            "LOGS_DIR": os.environ.get("LOGS_DIR", "/app/logs"),
            "SCHEMA_DIR": os.environ.get("SCHEMA_DIR", "/app/schemas"),
            "DATA_DIR": os.environ.get("DATA_DIR", "/app/data"),
        }
        
        print("[CONFIG] Environment configuration:")
        for key, value in config.items():
            if "KEY" not in key and "PASSWORD" not in key:  # Don't print secrets
                print(f"  {key}: {value}")
        
        return config
    
    @staticmethod
    def validate_all():
        """Run all validations."""
        print("[CONFIG] Starting configuration validation...")
        
        ConfigValidator.validate_paths()
        config = ConfigValidator.validate_environment()
        cred_issues = ConfigValidator.validate_credentials()
        
        if cred_issues:
            print("[CONFIG] Credential warnings:")
            for issue in cred_issues:
                print(f"  WARNING: {issue}")
        
        print("[CONFIG] Configuration validation complete.")
        return config


def get_config():
    """Get validated configuration."""
    return ConfigValidator.validate_all()
