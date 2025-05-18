"""Prompt library for managing and retrieving prompt templates."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llmforgekit.core.base import PromptTemplate
from llmforgekit.core.errors import ConfigError, PromptError
from llmforgekit.core.logging import get_logger
from llmforgekit.services.prompt.templates import StringPromptTemplate

logger = get_logger("services.prompt.library")


class PromptVersion:
    """Version information for a prompt template."""
    
    def __init__(self, version: str, template: PromptTemplate, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a template version.
        
        Args:
            version: The version identifier (e.g., "1.0.0")
            template: The template object
            metadata: Optional metadata for this version
        """
        self.version = version
        self.template = template
        self.metadata = metadata or {}


class PromptLibrary:
    """Library for storing and retrieving prompt templates.
    
    The prompt library stores templates with version tracking and provides
    methods for retrieving, adding, and managing templates.
    """
    
    def __init__(self):
        """Initialize an empty prompt library."""
        # Format: {template_name: {version: PromptVersion}}
        self.templates: Dict[str, Dict[str, PromptVersion]] = {}
        # Track the latest version for each template
        self.latest_versions: Dict[str, str] = {}
        
        logger.debug("Initialized empty PromptLibrary")
    
    def add_template(
        self,
        name: str,
        template: Union[str, PromptTemplate],
        version: str = "1.0.0",
        template_type: str = "string",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptTemplate:
        """Add a template to the library.
        
        Args:
            name: The name of the template
            template: The template string or object
            version: The version of the template
            template_type: The type of template ("string" or "jinja")
            metadata: Optional metadata for this template version
            
        Returns:
            The added template object
            
        Raises:
            PromptError: If the template_type is not supported
        """
        # Convert string to template object if needed
        if isinstance(template, str):
            if template_type.lower() == "string":
                template_obj = StringPromptTemplate(template, template_id=name)
            elif template_type.lower() == "jinja":
                # Import here to avoid dependency issues
                from llmforgekit.services.prompt.templates import JinjaPromptTemplate
                template_obj = JinjaPromptTemplate(template, template_id=name)
            else:
                raise PromptError(f"Unsupported template type: {template_type}")
        else:
            template_obj = template
        
        # Initialize the template dict if needed
        if name not in self.templates:
            self.templates[name] = {}
        
        # Add the version
        version_obj = PromptVersion(version, template_obj, metadata)
        self.templates[name][version] = version_obj
        
        # Update latest version if needed
        if name not in self.latest_versions or self._compare_versions(version, self.latest_versions[name]) > 0:
            self.latest_versions[name] = version
        
        logger.info(f"Added template '{name}' version {version} to library")
        return template_obj
    
    def get_template(
        self,
        template_name: str,
        version: Optional[str] = None,
    ) -> Optional[PromptTemplate]:
        """Get a template from the library.
        
        Args:
            template_name: The name of the template
            version: The version to retrieve. If None, the latest version is returned.
            
        Returns:
            The template, or None if not found
        """
        if template_name not in self.templates:
            logger.warning(f"Template '{template_name}' not found in library")
            return None
        
        # Use latest version if not specified
        if version is None:
            version = self.latest_versions.get(template_name)
            if not version:
                logger.warning(f"No latest version found for template '{template_name}'")
                return None
        
        # Get the specific version
        if version not in self.templates[template_name]:
            logger.warning(f"Version {version} of template '{template_name}' not found")
            return None
        
        logger.debug(f"Retrieved template '{template_name}' version {version}")
        return self.templates[template_name][version].template
    
    def format_prompt(
        self,
        template_name: str,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Format a prompt from the library.
        
        Args:
            template_name: The name of the template
            version: The version to use. If None, the latest version is used.
            **kwargs: Variables to insert into the template
            
        Returns:
            The formatted prompt, or None if the template is not found
            
        Raises:
            PromptError: If formatting fails
        """
        template = self.get_template(template_name, version)
        
        if template is None:
            return None
        
        try:
            return template.format(**kwargs)
        except PromptError as e:
            # Add template info to the error
            e.details = e.details or {}
            e.details.update({
                "template_name": template_name,
                "template_version": version or self.latest_versions.get(template_name, "unknown"),
            })
            raise
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates in the library.
        
        Returns:
            A list of dictionaries with template information
        """
        result = []
        for name, versions in self.templates.items():
            latest_version = self.latest_versions.get(name)
            template_info = {
                "name": name,
                "versions": list(versions.keys()),
                "latest_version": latest_version,
            }
            result.append(template_info)
        return result
    
    def load_from_directory(self, directory: Union[str, Path]) -> int:
        """Load templates from a directory.
        
        The directory should contain JSON files with template definitions.
        Each file should be in the format:
        {
            "name": "template_name",
            "version": "1.0.0",
            "type": "string",
            "template": "This is a $variable template",
            "metadata": {
                "description": "Template description",
                "author": "Author name",
                ...
            }
        }
        
        Args:
            directory: The directory path
            
        Returns:
            The number of templates loaded
            
        Raises:
            ConfigError: If directory doesn't exist or files are invalid
        """
        dir_path = Path(directory).expanduser()
        if not dir_path.exists() or not dir_path.is_dir():
            raise ConfigError(f"Template directory not found: {directory}")
        
        loaded_count = 0
        for file_path in dir_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Validate required fields
                if "name" not in data or "template" not in data:
                    logger.warning(f"Skipping invalid template file: {file_path} (missing required fields)")
                    continue
                
                # Add the template
                self.add_template(
                    name=data["name"],
                    template=data["template"],
                    version=data.get("version", "1.0.0"),
                    template_type=data.get("type", "string"),
                    metadata=data.get("metadata"),
                )
                loaded_count += 1
                
            except (json.JSONDecodeError, PromptError) as e:
                logger.error(f"Error loading template from {file_path}: {e}")
        
        logger.info(f"Loaded {loaded_count} templates from directory: {directory}")
        return loaded_count
    
    def save_to_directory(self, directory: Union[str, Path]) -> int:
        """Save templates to a directory.
        
        Args:
            directory: The directory path
            
        Returns:
            The number of templates saved
            
        Raises:
            ConfigError: If directory doesn't exist or can't be created
        """
        dir_path = Path(directory).expanduser()
        dir_path.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for name, versions in self.templates.items():
            for version_id, version_obj in versions.items():
                template = version_obj.template
                
                # Get the template text
                if hasattr(template, "template_text"):
                    template_text = template.template_text
                else:
                    logger.warning(f"Cannot save template '{name}' version {version_id} (no template_text attribute)")
                    continue
                
                # Determine template type
                template_type = "string"
                if "JinjaPromptTemplate" in template.__class__.__name__:
                    template_type = "jinja"
                elif "DynamicPromptTemplate" in template.__class__.__name__:
                    template_type = "dynamic"
                
                # Create data to save
                data = {
                    "name": name,
                    "version": version_id,
                    "type": template_type,
                    "template": template_text,
                    "metadata": version_obj.metadata,
                }
                
                # Save to file
                file_name = f"{name}_v{version_id.replace('.', '_')}.json"
                file_path = dir_path / file_name
                
                try:
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=2)
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving template to {file_path}: {e}")
        
        logger.info(f"Saved {saved_count} template versions to directory: {directory}")
        return saved_count

    @staticmethod
    def _compare_versions(version1: str, version2: str) -> int:
        """Compare two version strings.
        
        Args:
            version1: The first version
            version2: The second version
            
        Returns:
            1 if version1 > version2, -1 if version1 < version2, 0 if equal
        """
        v1_parts = [int(x) for x in version1.split(".")]
        v2_parts = [int(x) for x in version2.split(".")]
        
        # Pad with zeros if needed
        while len(v1_parts) < len(v2_parts):
            v1_parts.append(0)
        while len(v2_parts) < len(v1_parts):
            v2_parts.append(0)
        
        # Compare part by part
        for i in range(len(v1_parts)):
            if v1_parts[i] > v2_parts[i]:
                return 1
            elif v1_parts[i] < v2_parts[i]:
                return -1
        
        # Equal
        return 0