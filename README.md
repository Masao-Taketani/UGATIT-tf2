# U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

## Prepare Data
  1. Download the dataset from the [link](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing) and place them under `dataset` directory.
  2. Convert data into TFRecords.
```
python data_handler/tfrecords.py
```

## References
- Paper
  - [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)<br>

- Repos
  - [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch)
  - [taki0112/UGATIT](https://github.com/taki0112/UGATIT)

- Blogs
  - [Batch Normalization, Instance Normalization, Layer Normalization: Structural Nuances](https://becominghuman.ai/all-about-normalization-6ea79e70894b)