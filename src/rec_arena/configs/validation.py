"""Configuration validation utilities."""

from typing import Any, Dict, List, Union


class ConfigValidator:
    """Validates model configurations."""

    @staticmethod
    def validate_base_config(config: Dict[str, Any]) -> None:
        """Validate base configuration parameters."""
        required_fields = ["vocab_size", "embedding_dim"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

            value = config[field]
            try:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"{field} must be positive")
            except (TypeError, ValueError):
                raise ValueError(f"{field} must be a positive integer, got {value}")

    @staticmethod
    def validate_sequential_config(config: Dict[str, Any]) -> None:
        """Validate sequential model configuration."""
        ConfigValidator.validate_base_config(config)

        if "max_seq_length" in config:
            if (
                not isinstance(config["max_seq_length"], int)
                or config["max_seq_length"] <= 0
            ):
                raise ValueError("max_seq_length must be a positive integer")

        if "loss_type" in config:
            valid_losses = ["cross_entropy", "bce", "bpr", "sampled_softmax", "gbce"]
            if config["loss_type"] not in valid_losses:
                raise ValueError(
                    f"Invalid loss_type for sequential model: {config['loss_type']}"
                )

    @staticmethod
    def validate_implicit_config(config: Dict[str, Any]) -> None:
        """Validate implicit model configuration."""
        required_fields = ["num_users", "num_items", "embedding_dim"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

            value = config[field]
            try:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"{field} must be positive")
            except (TypeError, ValueError):
                raise ValueError(f"{field} must be a positive integer, got {value}")

        if "loss_type" in config:
            valid_losses = ["bce", "bpr"]
            if config["loss_type"] not in valid_losses:
                raise ValueError(
                    f"Invalid loss_type for implicit model: {config['loss_type']}"
                )


def validate_config(config: Union[Dict[str, Any], Any], model_type: str) -> None:
    """Validate configuration based on model type."""
    # Convert config object to dict if needed
    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        # Skip validation for unknown config types
        return

    if model_type == "sequential":
        ConfigValidator.validate_sequential_config(config_dict)
    elif model_type == "implicit":
        ConfigValidator.validate_implicit_config(config_dict)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
