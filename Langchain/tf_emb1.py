from typing import Any, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra


class TensorflowHubEmbeddings(BaseModel, Embeddings):
    """TensorflowHub embedding models.

    To use, you should have the ``tensorflow_text`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import TensorflowHubEmbeddings
            url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            tf = TensorflowHubEmbeddings(model_url=url)
    """

    embed: Any  #: :meta private:
    model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/3"
    """Model name to use."""

    def load(self, **kwargs: Any):
        """Initialize the tensorflow_hub and tensorflow_text."""
        super().__init__(**kwargs)
        try:
            import tensorflow_hub
        except ImportError:
            raise ImportError(
                "Could not import tensorflow-hub python package. "
                "Please install it with `pip install tensorflow-hub``."
            )

        self.embed = tensorflow_hub.load(self.model_url)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a TensorflowHub embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.embed(texts)['outputs'].numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a TensorflowHub embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.embed([text])['outputs'].numpy()[0]
        return embedding.tolist()