from abc import ABC, abstractmethod

from PIL import Image


class BaseProcessor(ABC):
    """All image processors implement this interface."""

    name: str = "base"

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and register in ModelManager."""
        ...

    @abstractmethod
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process image and return result."""
        ...

    def _ensure_model_loaded(self) -> None:
        from app.core.model_manager import ModelManager

        if not ModelManager.is_loaded(self.name):
            self.load_model()
