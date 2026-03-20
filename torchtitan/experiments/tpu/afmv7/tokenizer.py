"""AFMTextV7 tokenizer wrapper for TorchTitan."""

import os

from torchtitan.components.tokenizer import BaseTokenizer, build_hf_tokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


class AFMTokenizerWrapper(BaseTokenizer):
    """Wraps TAMM's AFMTokenizer as a TorchTitan BaseTokenizer.

    AFMTokenizer is a SentencePiece-based tokenizer with a text processor
    that converts newline characters to the special token ``<n>``.

    Args:
        vocab_path: Path to the SentencePiece ``.model`` vocabulary file.
    """

    def __init__(self, vocab_path: str) -> None:
        super().__init__()
        from tamm.tokenizers.afm import AFMTokenizer

        self._tok = AFMTokenizer(vocab_path=vocab_path)
        # BaseTokenizer expects eos_id as a plain int attribute.
        self.eos_id = self._tok.eos_id

    def encode(self, *args, **kwargs) -> list[int]:
        """Encode text to token IDs.

        Passes through all arguments to AFMTokenizer.encode(), which
        delegates to SentencePiece and supports ``add_bos`` and ``add_eos``
        keyword arguments used by the torchtitan dataloader.
        """
        result = self._tok.encode(*args, **kwargs)
        # SentencePiece returns a list for a single string input.
        return result

    def decode(self, *args, **kwargs) -> str:
        return self._tok.decode(*args, **kwargs)

    def get_vocab_size(self) -> int:
        return len(self._tok)


def build_afm_tokenizer(job_config: JobConfig) -> BaseTokenizer:
    """Build a tokenizer from job_config.model.hf_assets_path.

    The path should point to either:
    - A directory containing a ``*.model`` SentencePiece vocabulary file, or
    - Directly to the ``.model`` file itself.

    If the path is a directory containing a HuggingFace ``tokenizer.json``
    (e.g. the built-in ``tests/assets/tokenizer`` for debug runs) but no
    ``.model`` file, falls back to ``build_hf_tokenizer``.
    """
    path = job_config.model.hf_assets_path
    if _is_hf_tokenizer_dir(path):
        logger.info(f"No SentencePiece .model found in {path}; falling back to HF tokenizer")
        return build_hf_tokenizer(job_config)
    vocab_path = _resolve_vocab_path(path)
    logger.info(f"Loading AFM tokenizer from {vocab_path}")
    return AFMTokenizerWrapper(vocab_path=vocab_path)


def _is_hf_tokenizer_dir(path: str) -> bool:
    """Return True if path is a directory containing a HuggingFace tokenizer.json."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "tokenizer.json"))


def _resolve_vocab_path(path: str) -> str:
    """Return the SentencePiece .model file path.

    If ``path`` is a directory, searches for a single ``.model`` file inside.
    If ``path`` already ends in ``.model``, returns it directly.
    """
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        candidates = [f for f in os.listdir(path) if f.endswith(".model")]
        if len(candidates) == 1:
            return os.path.join(path, candidates[0])
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple .model files found in {path}: {candidates}. "
                "Set hf_assets_path to the specific file."
            )
        raise FileNotFoundError(
            f"No SentencePiece .model file found in directory: {path}"
        )

    raise FileNotFoundError(
        f"AFM tokenizer path not found: {path}. "
        "Set --model.hf_assets_path to a directory containing a .model file "
        "or directly to the .model file."
    )
