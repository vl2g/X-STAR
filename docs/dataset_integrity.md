# benchmark integrity

The released results use the fixed X-STAR split: 4,515 training, 645 validation and 1,289 test records. Record IDs and filenames are disjoint across splits; image contents are not fully disjoint.

A byte-level audit of all 6,449 image files found 61 identical-image pairs: 34 within training, two within test, and 25 across splits. Fifteen test images and eight validation images match a training image; two additional pairs span validation and test. No record assignments were changed for the reported benchmark. This overlap is a limitation; its effect on the reported results has not been measured.

Two pairs of identical test images produce tied scores. Evaluation retains the fixed gallery order for ties. Recall is macro-averaged over all 13 query sets, each scored against the complete 1,289-image gallery. See the [evaluation report](evaluation/main_checkpoint_cached_features.md) for detailed counts and tie-policy bounds.

These observations concern byte-identical image files. Different-angle photographs remain distinct records. No scene-disjointness claim is made.
