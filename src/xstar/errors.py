"""Exception types, kept free of heavy imports so the CLIs can catch them cheaply."""


class DataError(ValueError):
    """Missing, malformed or misaligned dataset or feature input."""


class ConfigError(ValueError):
    """Invalid or incomplete configuration."""


class CheckpointError(RuntimeError):
    """Checkpoint missing, malformed or incompatible with the model."""


class OutputSafetyError(RuntimeError):
    """An output location could overwrite or sit inside an input."""


class EncoderError(RuntimeError):
    """CLIP encoder unavailable (package or weights missing)."""
