"""
App package initialization.
Handles configuration validation and module setup.
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration will be initialized at app startup
_config = None

def initialize():
    """Initialize app configuration."""
    global _config
    try:
        from ..api.config import get_config
        _config = get_config()
        logger.info("App initialization complete")
    except Exception as e:
        logger.warning(f"Configuration initialization warning: {e}")

def get_config():
    """Get configuration."""
    return _config
