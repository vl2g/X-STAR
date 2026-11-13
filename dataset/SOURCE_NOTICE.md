# X-STAR annotation release

Creators: Dikshant Sharma, Prateek and Anand Mishra, Indian Institute of Technology Jodhpur. We acknowledge the National Language Translation Mission (Bhashini).

X-STAR-created annotation contributions, translations, split lists and mapping are released under CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/. Give attribution, retain the license and source notices, identify changes, and share adaptations under the applicable license. Third-party source material retains its own terms; no rights over source photographs are granted.

## Source attribution

- Bharat Scene Text Dataset (5,949 image records): https://huggingface.co/datasets/Bhashini-IITJ/Bharat-Scene-Text-Dataset and https://github.com/Bhashini-IITJ/BharatSceneTextDataset. Its annotation terms are CC BY-SA 4.0; photographs retain per-image licenses. OCR lines for these records were produced upstream using IndicPhotoOCR and translated to English. This release does not contain source photographs.
- RCTW-17 (500 image records): Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu and Xiang Bai, “ICDAR2017 Competition on Reading Chinese Text in the Wild (RCTW-17),” ICDAR 2017. https://rctw.vlrlab.net/dataset. Chinese scene-text inputs derive from supplied line annotations and upstream English translations. No named RCTW license is asserted; STR-Fewer-Labels reports author permission for commercial and non-commercial use: https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md#license-of-datasets. X-STAR's license applies to our contributions and does not replace upstream terms.
- IndicPhotoOCR: https://github.com/Bhashini-IITJ/IndicPhotoOCR.

## Contents and changes

6,449 records, 13 native query-language fields, 12 back-translated English fields, supplied English OCR lines, fixed train/validation/test assignments, and three ordered split lists. There are 4,515 training, 645 validation and 1,289 test records. No images, feature arrays, credentials or training checkpoints are included.

Only server-specific `abs_path` fields and the metadata `root` were removed for portability; license/source metadata was added. Image IDs, relative filenames, all text fields, record order and split assignments are unchanged. The manifest file hash therefore differs from the training-input manifest; `provenance.json` records both hashes. Use `xstar_annotations.json` as `dataset.manifest` and the `splits/` files in `configs/paths.yaml`.

The split has disjoint IDs and filenames, with known byte-identical image contents across splits. See the repository's dataset-integrity documentation. Query generation and machine translations can contain errors. TextCaps data, pretrained feature caches and source images must be acquired separately.
