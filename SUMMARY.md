**Severstal: Steel Defect Detection 2019 Challenge** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the surface defect detection domain. 

The dataset consists of 18074 images with 19958 labeled objects belonging to 4 different classes including *defect_3*, *defect_1*, *defect_4*, and other: *defect_2*.

Images in the Severstal dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 11408 (63% of the total) unlabeled images (i.e. without annotations). There are 2 splits in the dataset: *test* (5506 images) and *train* (12568 images). The dataset was released in 2019 by the Severstal, Russia.

Here is the visualized example grid with animated annotations:

[animated grid](https://github.com/dataset-ninja/severstal/raw/main/visualizations/horizontal_grid.webm)
