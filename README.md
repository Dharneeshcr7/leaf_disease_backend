# leaf_disease_dection_pytorch

For Dataset refer plant village dataset in kaggle:
https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated


Overview of Leaf disease classification Model used:

   1.Extract features of the input image from efficientnetb0 model at global average pooling layer.
   
   2.Apply SVM on extracted features to classify the leaves based on types of diseases.

   3.Use Unet for severity estimation of leaf by segmenting disease area, background and leaf in image.

Overview of Severity estimation Model used:

  1.Segment leaf image into leaf,background,disease area using U-Net.

  2.Use pixel counting method to estimate severity of disease.


