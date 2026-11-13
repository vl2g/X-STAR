# Licensing and attribution

The [MIT license](../LICENSE) covers this project's own code, configuration and software documentation. Source photographs, upstream annotations, feature caches and paper figures retain separate terms. Our released weights use [MIT](../MODEL_LICENSE.txt); our annotation contributions use [CC BY-SA 4.0](../dataset/LICENSE-ANNOTATIONS.txt). The SVG header uses outlined Liberation lettering; its [font notice](assets/brand/liberation-font-license.txt) is retained.

## Source datasets

- **Bharat Scene Text Dataset:** source photographs retain mixed, per-image licenses. Follow the [official dataset card](https://huggingface.co/datasets/Bhashini-IITJ/Bharat-Scene-Text-Dataset) and image source records. Do not infer photograph rights from a software repository license.
- **RCTW-17:** the [official dataset page](https://rctw.vlrlab.net/dataset) provides downloads without an explicit named license. The [STR-Fewer-Labels licensing table](https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md#license-of-datasets) reports dataset-author email permission for commercial and non-commercial use. This is reported correspondence, not an MIT or Creative Commons license.
- **TextCaps:** obtain data under its [upstream terms](https://textvqa.org/textcaps/dataset/) and preserve the terms of its source photographs. Raw TextCaps preparation is not implemented here.

Our annotation release is CC BY-SA 4.0 and retains source notices; the main model weights are MIT licensed. Feature-cache release terms remain unspecified. The included Chinese-source mapping records filenames and verification hashes, not source photographs or OCR transcripts.

## Software acknowledgements

We thank [AI4Bharat](https://ai4bharat.iitm.ac.in/) for [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2), used for Indic-language translation during dataset preparation.

[OpenAI CLIP](https://github.com/openai/CLIP), [IndicPhotoOCR](https://github.com/Bhashini-IITJ/IndicPhotoOCR), and the [ColBERT](https://github.com/stanford-futuredata/ColBERT) late-interaction design are acknowledged. Dependencies retain their own licenses; OpenAI CLIP is installed separately. The [project page's image credits](https://vl2g.github.io/X-STAR/image-credits.html) cover its selected photographs.

We acknowledge the **National Language Translation Mission ([Bhashini](https://bhashini.gov.in/))** for supporting this project at Indian Institute of Technology Jodhpur.

See [source attribution](../dataset/SOURCE_NOTICE.md), [annotation provenance](../dataset/annotation_release.json) and [downloads](resources.md).
