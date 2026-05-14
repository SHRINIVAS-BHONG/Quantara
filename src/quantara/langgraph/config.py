"""
Configuration settings for the LangGraph Quantara trading system.

This module provides configuration management for the LangGraph-based
trading system, including API settings, model configurations, and
system parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LangGraphConfig:
    """
    Configuration class for LangGraph trading system.
    """
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    
    # API Configuration
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    
    # Market Data APIs
    polymarket_api_key: Optional[str] = None
    kalshi_api_key: Optional[str] = None
    apify_api_key: Optional[str] = None
    
    # System Configuration
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    
    # Checkpointing Configuration
    checkpoint_enabled: bool = True
    checkpoint_storage: str = "memory"  # "memory", "redis", "postgres"
    checkpoint_retention_hours: int = 24
    
    # Risk Management
    default_risk_tolerance: float = 0.5
    max_portfolio_risk: float = 0.7
    max_individual_risk: float = 0.8
    max_correlation: float = 0.6
    min_diversification: float = 0.3
    
    # Performance Thresholds
    min_trader_confidence: float = 0.6
    min_data_quality: float = 0.7
    max_analysis_time: int = 300  # seconds
    
    # Monitoring Configuration
    enable_monitoring: bool = True
    monitoring_interval: int = 300  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "volatility": 0.15,
        "sentiment_shift": 0.3,
        "risk_violation": 0.8
    })
    
    # Logging Configuration
    log_level: str = "INFO"
    structured_logging: bool = True
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        # Load API keys from environment
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = self.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.polymarket_api_key = self.polymarket_api_key or os.getenv("POLYMARKET_API_KEY")
        self.kalshi_api_key = self.kalshi_api_key or os.getenv("KALSHI_API_KEY")
        self.apify_api_key = self.apify_api_key or os.getenv("APIFY_API_KEY")
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Validate risk tolerance ranges
        if not (0.0 <= self.default_risk_tolerance <= 1.0):
            raise ValueError("default_risk_tolerance must be between 0.0 and 1.0")
        
        if not (0.0 <= self.max_portfolio_risk <= 1.0):
            raise ValueError("max_portfolio_risk must be between 0.0 and 1.0")
        
        # Validate temperature range
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        
        # Validate positive integers
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "retry_attempts": self.retry_attempts,
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_storage": self.checkpoint_storage,
            "default_risk_tolerance": self.default_risk_tolerance,
            "max_portfolio_risk": self.max_portfolio_risk,
            "enable_monitoring": self.enable_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "alert_thresholds": self.alert_thresholds,
            "log_level": self.log_level,
            "structured_logging": self.structured_logging
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LangGraphConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "LangGraphConfig":
        """Create configuration from environment variables."""
        return cls(
            llm_provider=os.getenv("LANGGRAPH_LLM_PROVIDER", "openai"),
            llm_model=os.getenv("LANGGRAPH_LLM_MODEL", "gpt-4"),
            llm_temperature=float(os.getenv("LANGGRAPH_LLM_TEMPERATURE", "0.1")),
            max_concurrent_requests=int(os.getenv("LANGGRAPH_MAX_CONCURRENT", "10")),
            request_timeout=int(os.getenv("LANGGRAPH_REQUEST_TIMEOUT", "30")),
            checkpoint_enabled=os.getenv("LANGGRAPH_CHECKPOINT_ENABLED", "true").lower() == "true",
            checkpoint_storage=os.getenv("LANGGRAPH_CHECKPOINT_STORAGE", "memory"),
            default_risk_tolerance=float(os.getenv("LANGGRAPH_RISK_TOLERANCE", "0.5")),
            enable_monitoring=os.getenv("LANGGRAPH_ENABLE_MONITORING", "true").lower() == "true",
            log_level=os.getenv("LANGGRAPH_LOG_LEVEL", "INFO")
        )


@dataclass
class UserPreferences:
    """
    User preferences for trading analysis.
    """
    risk_tolerance: float = 0.5
    preferred_platforms: list = field(default_factory=lambda: ["polymarket", "kalshi"])
    niche_preference: str = ""
    max_portfolio_risk: float = 0.7
    max_individual_risk: float = 0.8
    enable_monitoring: bool = True
    update_interval: int = 300
    alert_channels: list = field(default_factory=lambda: ["email"])
    
    def __post_init__(self):
        """Validate user preferences."""
        if not (0.0 <= self.risk_tolerance <= 1.0):
            raise ValueError("risk_tolerance must be between 0.0 and 1.0")
        
        if not (0.0 <= self.max_portfolio_risk <= 1.0):
            raise ValueError("max_portfolio_risk must be between 0.0 and 1.0")
        
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")


# Global configuration instance
_config: Optional[LangGraphConfig] = None


def get_config() -> LangGraphConfig:
    """
    Get the global configuration instance.
    
    Returns:
        LangGraphConfig instance
    """
    global _config
    if _config is None:
        _config = LangGraphConfig.from_env()
    return _config


def set_config(config: LangGraphConfig):
    """
    Set the global configuration instance.
    
    Args:
        config: LangGraphConfig instance to set as global
    """
    global _config
    _config = config


def create_user_preferences(**kwargs) -> UserPreferences:
    """
    Create user preferences with optional overrides.
    
    Args:
        **kwargs: Preference overrides
        
    Returns:
        UserPreferences instance
    """
    return UserPreferences(**kwargs)