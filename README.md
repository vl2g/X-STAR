<p align="center">
  <img src="docs/assets/brand/xstar-header.svg" width="100%" alt="X-STAR">
</p>
<h3 align="center">Cross-Lingual Scene Text-Aware Image Retrieval</h3>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Pending-B31B1B?style=flat-square)](docs/resources.md#paper)
[![Project Page](https://img.shields.io/badge/Project-Page-6C63FF?style=flat-square)](https://vl2g.github.io/projects/X-STAR/)
[![Dataset](https://img.shields.io/badge/Dataset-Download-008C95?style=flat-square)](docs/resources.md#dataset)
[![Weights](https://img.shields.io/badge/Weights-Download-DA7C18?style=flat-square)](docs/resources.md#checkpoints)
[![Docs](https://img.shields.io/badge/Docs-Read%20now-007BFF?style=flat-square)](docs/README.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
[![Code License](https://img.shields.io/badge/Code%20license-MIT-78A200?style=flat-square)](LICENSE)

</div>

---

**X-STAR** is a benchmark for cross-lingual scene text-aware image retrieval, with **6,449 images and queries in 13 languages**. Our model, **CoLITE-CLIP**, combines visual features and scene text to retrieve images across languages.

The code supports feature extraction, training, evaluation and retrieval. OCR recognition and translation are upstream preprocessing; supplied English OCR lines and translated queries are the inputs.

## Quick start

Python 3.10+; run from this repository inside a virtual environment:

```bash
git clone https://github.com/vl2g/X-STAR.git
cd X-STAR
pip install -e .
cp configs/paths.example.yaml configs/paths.yaml
# Fill in your manifest, cached-feature and checkpoint paths.
python scripts/retrieve.py --paths configs/paths.yaml \
  --checkpoint xstar_main --query-image-id A_image_103 \
  --query-set hindi_to_english --top-k 5
```

Download the [annotations](https://github.com/vl2g/X-STAR/releases/download/v0.1.0/xstar_annotations.zip) and [model weights](https://github.com/vl2g/X-STAR/releases/download/v0.1.0/colite_clip_xstar.pt). Source images and cached features are acquired or prepared separately. For extraction, free-text queries and GPU setup, see [installation and usage](docs/usage.md).

## Results

Results on the **X-STAR test split**. Recall@K (%), macro-averaged over 13 query sets; 1,289 gallery images and 16,757 queries per experiment.

| Evaluation | Model | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| Cached-feature evaluation | **CoLITE-CLIP** | **46.5** | **65.8** | **73.0** |
| Cached-feature evaluation | CLIP ViT-B/32 | 24.6 | 42.8 | 52.0 |

The split contains identical images across splits. Evaluation uses fixed gallery-order ties. [Reports and reproduction limits →](docs/reproduction.md)

## Explore

- [Dataset and preparation](dataset/README.md) — source images, query fields and feature formats.
- [Method and benchmark](docs/method.md) — CoLITE-CLIP and all 13 query languages.
- [Train, evaluate and retrieve](docs/usage.md) — commands and configuration.
- [Project website](https://vl2g.github.io/projects/X-STAR/) — method, results and retrieval examples.
- [Resources](docs/resources.md) — availability of paper, dataset and checkpoints.

## License and citation

Code is [MIT licensed](LICENSE), and the released weights use [MIT](MODEL_LICENSE.txt). Our annotation contributions use [CC BY-SA 4.0](dataset/LICENSE-ANNOTATIONS.txt). Source photographs and third-party material retain their source terms; see [licensing](docs/licensing.md). Paper and citation metadata are [pending](docs/resources.md#citation).

## Acknowledgements

We thank [AI4Bharat](https://ai4bharat.iitm.ac.in/) for [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2), used for Indic-language translation during dataset preparation.

We acknowledge the **National Language Translation Mission ([Bhashini](https://bhashini.gov.in/))** for supporting this project at Indian Institute of Technology Jodhpur.

Built on [Bharat Scene Text Dataset](https://github.com/Bhashini-IITJ/BharatSceneTextDataset), [RCTW-17](https://rctw.vlrlab.net/), [IndicPhotoOCR](https://github.com/Bhashini-IITJ/IndicPhotoOCR), [OpenAI CLIP](https://github.com/openai/CLIP) and the [ColBERT](https://github.com/stanford-futuredata/ColBERT) late-interaction design; pre-trained on [TextCaps](https://textvqa.org/textcaps/).
