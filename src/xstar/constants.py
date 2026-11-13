"""Names and sizes shared across the package."""

CLIP_MODEL_NAME = "ViT-B/32"
CLIP_DIM = 512
VISUAL_TOKENS = 50          # CLS + 7x7 patches for ViT-B/32 at 224x224
TEXT_CONTEXT = 77           # CLIP text context length

SPLITS = ("train", "val", "test")

# The 12 languages whose queries are machine-translated back to English.
TRANSLATED_LANGUAGES = (
    "assamese", "bengali", "gujarati", "hindi", "kannada", "malayalam",
    "marathi", "odia", "punjabi", "tamil", "telugu", "chinese",
)

# The 13 evaluated query sets, in the order used by the original evaluator.
# Each set is a manifest field holding English (or back-translated English) text.
QUERY_SETS = ("english_query",) + tuple(f"{lang}_to_english" for lang in TRANSLATED_LANGUAGES)

QUERY_SET_LANGUAGE = {"english_query": "english"}
QUERY_SET_LANGUAGE.update({f"{lang}_to_english": lang for lang in TRANSLATED_LANGUAGES})

# Native-language query fields (not encoded by CoLITE-CLIP or the CLIP baseline).
NATIVE_QUERY_FIELDS = ("english_query",) + tuple(f"{lang}_query" for lang in TRANSLATED_LANGUAGES)

CHINESE_FOLDER = "N"

# File format written by scripts/extract_ocr_features.py.
OCR_FORMAT = "xstar-ocr-lines-v1"
