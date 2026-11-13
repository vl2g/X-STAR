# Method and benchmark

## Overview

**X-STAR** asks whether an image containing scene text in one language can be
retrieved with a natural-language query written in another. Given a gallery of
street-scene images whose text spans Indian languages, English and Chinese,
and a query in any of 13 languages, the goal is to rank the matching image at
the top.

**CoLITE-CLIP** (Co-attentive Late Interaction for Text-Image Embedding over
CLIP) reduces the cross-lingual problem to a shared English space through
machine translation and then:

1. represents each image by 50 frozen CLIP ViT-B/32 visual tokens
   (CLS + 49 patches, projected to 512-d);
2. represents its scene text by one CLIP text embedding per English OCR line
   (up to K = 40 lines, with a learned null token for padding);
3. fuses the two with 11 blocks of bidirectional visual/OCR cross-attention;
4. scores a query against the fused 90-token bank with ColBERT-style MaxSim
   over the query's per-token CLIP text embeddings;
5. trains with a symmetric InfoNCE loss over in-batch negatives, first on
   TextCaps and then on X-STAR.


## Benchmark at a glance

| | |
|---|---|
| Images | 6,449 (4,515 train / 645 validation / 1,289 test) |
| Image sources | Bharat Scene Text Dataset (5,949 images) and RCTW-17 (500 Chinese images) |
| Query languages | 13: English, Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Chinese |
| Queries | 83,837 overall; 16,757 in the test split |
| Evaluation | Text-to-image retrieval over the full 1,289-image test gallery, Recall@1/5/10 per query language and macro-averaged |

These counts were re-verified against the local final manifest by
[`tools/audit_release_inputs.py`](../tools/audit_release_inputs.py).
Dataset details, source-image acquisition and field definitions are in
[`dataset/README.md`](../dataset/README.md).
