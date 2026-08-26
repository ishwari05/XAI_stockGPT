from typing import Dict, Type, Any
from .base import BasePredictionModel


class ModelRegistry:
    """
    Central Model Registry.
    Enables plug-and-play addition of new predictive models purely through configuration.
    """

    _registry: Dict[str, Type[BasePredictionModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: Type[BasePredictionModel]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BasePredictionModel]:
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(
                f"Model '{name}' not found in registry. Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BasePredictionModel:
        model_cls = cls.get(name)
        return model_cls(config)

    @classmethod
    def list_models(cls) -> list:
        return list(cls._registry.keys())
