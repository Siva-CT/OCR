class BaseOCR:
    def extract_text(self, image) -> str:
        """
        Extracts text from the provided image.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Subclasses must implement extract_text")