"""Configuration management for LLMForgeKit."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import pydantic


class LLMForgeKitConfig(pydantic.BaseModel):
    """Configuration for LLMForgeKit."""
    
    # Default paths
    config_dir: Path = Path(os.environ.get("LLMFORGEKIT_CONFIG_DIR", "~/.llmforgekit")).expanduser()
    
    # LLM Provider configs
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # LLM default parameters
    default_temperature: float = 0.7
    default_max_tokens: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> "LLMForgeKitConfig":
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "LLMForgeKitConfig":
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            A new configuration object with values from the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid JSON
        """
        config_path = Path(config_path).expanduser()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Read the file
        with open(config_path, "r") as f:
            try:
                # Parse JSON content
                config_data = json.load(f)
                
                # Create a new config object with the loaded data
                return cls(**config_data)
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file: {e}")
    
    @classmethod
    def find_and_load_config(cls) -> "LLMForgeKitConfig":
        """Find and load configuration from standard locations.
        
        Looks for configuration files in the following locations:
        1. ./llmforgekit.json (current directory)
        2. ~/.llmforgekit/config.json (user's home directory)
        3. /etc/llmforgekit/config.json (system-wide)
        
        If no configuration files are found, falls back to environment variables.
        
        Returns:
            A configuration object
        """
        # Define possible config locations
        locations = [
            Path("./llmforgekit.json"),  # Current directory
            Path("~/.llmforgekit/config.json").expanduser(),  # User's home directory
            Path("/etc/llmforgekit/config.json"),  # System-wide config
        ]
        
        # Try each location
        for location in locations:
            if location.exists():
                try:
                    return cls.from_file(str(location))
                except Exception as e:
                    print(f"Warning: Failed to load config from {location}: {e}")
        
        # Fall back to environment variables if no files found
        return cls.from_env()
    
    def save_to_file(self, config_path: str) -> None:
        """Save the current configuration to a JSON file.
        
        Args:
            config_path: Path where the configuration should be saved
            
        Raises:
            OSError: If creating the directory or writing the file fails
        """
        config_path = Path(config_path).expanduser()
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary and handle Path objects
        config_dict = self.model_dump()  # Use model_dump() in Pydantic v2
        
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        # Save as JSON
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    
    def merge(self, other: "LLMForgeKitConfig") -> "LLMForgeKitConfig":
        """Merge with another configuration.
        
        Non-None values from the other configuration will override
        values in this configuration.
        
        Args:
            other: Another configuration object
            
        Returns:
            A new merged configuration object
        """
        data = self.dict()
        other_data = other.dict()
        
        # Only override values that are not None in other
        for key, value in other_data.items():
            if value is not None:
                data[key] = value
        
        return LLMForgeKitConfig(**data)


# Global config instance
# This will try to find a config file, and fall back to environment variables
config = LLMForgeKitConfig.find_and_load_config()