# test_config.py
"""Test script for the LLMForgeKit configuration system."""

import os
from llmforgekit.core.config import LLMForgeKitConfig

def test_basic_config():
    # Create a basic config
    config = LLMForgeKitConfig()
    print("Basic config:")
    print(f"Max retries: {config.max_retries}")
    print(f"Default temperature: {config.default_temperature}")

def test_env_config():
    # Set an environment variable
    os.environ["OPENAI_API_KEY"] = "test_key_from_env"
    
    # Load from environment
    config = LLMForgeKitConfig.from_env()
    print("\nConfig from environment:")
    print(f"OpenAI API key: {'Set' if config.openai_api_key else 'Not set'}")
    
    # Clean up
    del os.environ["OPENAI_API_KEY"]

def test_file_config():
    # Create a temporary config file
    import tempfile
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("""
        {
            "openai_api_key": "test_key_from_file",
            "max_retries": 5,
            "default_temperature": 0.9
        }
        """)
        temp_path = f.name
    
    # Load from file
    try:
        config = LLMForgeKitConfig.from_file(temp_path)
        print("\nConfig from file:")
        print(f"OpenAI API key: {'Set' if config.openai_api_key else 'Not set'}")
        print(f"Max retries: {config.max_retries}")
        print(f"Default temperature: {config.default_temperature}")
    finally:
        # Clean up
        import os
        os.unlink(temp_path)

def test_save_config():
    # Create a config
    config = LLMForgeKitConfig(
        openai_api_key="test_save_key",
        max_retries=7
    )
    
    # Save to temporary file
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_path = f"{temp_dir}/test_config.json"
    
    try:
        config.save_to_file(temp_path)
        print("\nSaved config to file:")
        
        # Read it back
        new_config = LLMForgeKitConfig.from_file(temp_path)
        print(f"Read back OpenAI API key: {'Set' if new_config.openai_api_key else 'Not set'}")
        print(f"Read back max retries: {new_config.max_retries}")
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_basic_config()
    test_env_config()
    test_file_config()
    test_save_config()