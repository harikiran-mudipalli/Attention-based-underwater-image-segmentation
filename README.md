# Attention based underwater image segmentation
## Abstarct
Semantic Segmentation is a popular problem in the computer vision domain which is used for estimating scene geometry, inferring interactions and spatial relationships among objects, salient object identification, etc., [1]. On a broader level, segmentation involves three steps, classifying certain objects in an image, localizing them, and grouping the pixels in the localized image by creating a segmentation mask [8].<p>
 
The recent introduction of transformer-based models and attention mechanisms in computer vision tasks has taken the spotlight. In this project, we are interested in applying the attention mechanism to a new segmentation model from [1] to underwater images. Semantic segmentation of underwater images is not well explored as in the case with terrestrial object segmentation. Authors of [1] have curated a dataset called Segmentation of Underwater IMagery (SUIM) and implemented SOTA segmentation models on it to compare with their model called SUIM-Net. We tweaked the SUIM-Net Residual Skip Block (RSB) by experimenting with the Efficient Channel Attention module [2] and Triplet Attention module [3] inside the SUIM-Net.<br>
 
## Architectures involved:
### SUIM-Net
 ![image](https://user-images.githubusercontent.com/43600130/167754214-e5e5cba7-895b-4a00-b3c4-5c36e1d18f9e.png)

 
## Experiment setup
• Batch size: 4<br>
• Epochs: 40<br>
• steps per epoch: 100<br>
• learning rate: 1e-4<br>
• optimizer: Adam<br>
• loss = 'binary_crossentropy’<br>
• GPU: Tesla T4 (Colab pro)<br>
*SUIM-Net + ECA - 18 + 22 + 40 epochs: There were breaks in the training of SUIM-Net
training. Though the plots of SUIM-Net + ECA look different, the results are identical to
the other two experiments but on a different scale.<br>

## References:
[1] [Semantic Segmentation of Underwater Imagery: Dataset and Benchmark](https://arxiv.org/pdf/2004.01241.pdf)<br>
[2] [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/pdf/1910.03151.pdf)<br>
[3] [Rotate to Attend: Convolutional Triplet Attention Module](https://arxiv.org/pdf/2010.03045.pdf)<br>
[4] [An Overview of the Attention Mechanisms in Computer Vision](https://iopscience.iop.org/article/10.1088/1742-6596/1693/1/012173/pdf)<br>
[5] [ECA-Net blog(code used for this project)](https://blog.paperspace.com/attention-mechanisms-in-computer-vision-ecanet/)<br>
[6] [Triplet Attention Explained (WACV 2021)](https://blog.paperspace.com/triplet-attention-wacv-2021/)<br>
[7] [Triplet Attention GitHub(used for this project)](https://github.com/jyhengcoder/Triplet-Attention-tf)<br>
[8] [Computer Vision Tutorial: A Step-by-Step Introduction to Image Segmentation Techniques (Part 1)](https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/)<br>
[9] [Dataset and Code of SUIM-Net used by authors in their paper(Used for this project)](http://irvlab.cs.umn.edu/image-segmentation/suim-and-suim-net)<br>
