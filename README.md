# [Cars images clusterization](https://www.kaggle.com/datasets/markhomeless/intellivision-case)
## [Skillfactory](https://skillfactory.ru) Data Science project

Sklearn.preprocessing, cluster, mixture, neighbors, manifold

matplotlib.pyplot, seaborn, mpl_toolkits.mplot3d.Axes3D, plotly.express, zipfile.ZipFile

:arrow_down: [read notebook](README.md###Read)
<hr>
<p> </p>

### Problem  

[IntelliVision](https://www.intelli-vision.com/) develops software for image processing and video analysis. One of the key projects of IntelliVision is Smart City/Transportation, a system that ensures road safety and more efficient operation of parking lots. 

CV (Computer Vision) is at the heart of all the listed features of the project. To implement them, the company uses models that use huge marked-up datasets with images of vehicles for training. But to work with the flow of new data, the company needs to automate labeling of additional parameters of the car in the image:

* type of car (body),
* the angle of the picture (rear/front view),
* the color of the car.

It is also necessary to automate search for outliers in the data (highlights and glare in images, images that do not have cars, etc.).

<center> <img src=https://i.ibb.co/hLcBpZF/2023-03-27-12-11-17.png align="right" width="500"/> </center>

**Hypothesis:** the markup of the new data can be effectively carried out using clustering methods. Intermediate feature representation (descriptors) obtained on convolutional layers of convolutional neural networks trained on various datasets and applied at various tasks from image classification by color to classification of vehicle types can be used as clustering data.

The [source folder](https://drive.google.com/file/d/1vkQaj0Lr4Jwkumli7k9IzCtxFP1tIoXH/view) with the data provided by the IntelliVision team has the following structure:

```
IntelliVision_case
├─descriptors
    └─efficientnet-b7.pickle
    └─osnet.pickle
    └─vdc_color.pickle
    └─vdc_type.pickle
├─row_data
    └─veriwild.zip
├─images_paths.csv 
```

* `efficientnet-b7.pickle` — descriptors allocated by the classification model with the [EfficientNet](https://medium.com/analytics-vidhya/efficientnet-the-state-of-the-art-in-imagenet-701d4304cfa3) architecture version 7 (convolutional neural network, pre-trained on the [ImageNet](https://ru.wikipedia.org/wiki/ImageNet) dataset with images of more than 1000 different classes). This model did not see the veriwiId dataset during training.

* `osnet.pickle` — descriptors allocated by the [OSNet](https://medium.com/@moncefboujou96/omni-scale-feature-learning-for-person-re-identification-6e09df1c9a1a) model trained to detect people, animals and vehicles. The model was not trained on the original veriwiId dataset.

* `vdc_color.pickle` — descriptors allocated by the regression model to determine the color of vehicles in RGB format. Partially trained on the original veriwild dataset.

* `vdc_type.pickle` — descriptors allocated by the vehicle classification model by type in ten classes. Partially trained on the original veriwild dataset.

* The `row_data` folder contains a zip archive with 416 314 original images of cars taken by road cameras from different angles.

* The `images_paths.csv` file contains a list of full paths to images.


**Task:** using the descriptors obtained, divide the images into clusters, interpret and compare the research results. An additional task is to discriminate outliers among the images.

**Business objective:** to investigate the possibility of using clustering algorithms to mark up new data and search for outliers.
<br>

### Read .ipynb

Jupyter notebook is too heavy to be rendered by GitHub. There are two alternatives to read it:

- read at [nbviewer](https://nbviewer.org/github/MapleBloom/Cars_clusterization/blob/main/Cars_clusterization_100.ipynb). At this case three interactive 3D-plots are not available;
- clone repository to read with all plots.
<br>

### Results

The descriptors data have been scaled by different scalers and the dimensionality of the data space has been reduced by `PCA`-method while preserving 90% of the information.

Images clustering was performed based on their descriptors using `KMeans`, `GaussianMixture` and `AgglomerativeClustering` with structuring through `neighbors.kneighbors_graph` algorithms. 

Internal metrics, `Calinski-Harabasz index` and `Davies-Bouldin index`, were used to select the optimal number of clusters and compare the algorithms. 

Interpretation of the obtained results was performed based on a random sample of images, which were read from the zip archive without unpacking with the help of `zipfile.ZipFile`. 

The clusters obtained were plotted in 2D and 3D-space using `t-SNE` for dimensionality reduction and `matplotlib.pyplot`, `seaborn`, `plotly.express`, `mpl_toolkits.mplot3d.Axes3D` for visualization. The best models show good density, consistency and clear cluster boundaries.

The `OPTICS` algorithm with `dbscan` method was applied to search for outliers. 

All modeling results were aggregated in a dictionary-like structure and exported to `pickle` format.

The best results at determining the body type and view angle were obtained on the base of `vdc_type` descriptor, and for color labeling - on the `vdc_color` descriptor.

Labels of different models coincides at the most. To label zones with ambiguous markup it could be effective take results of different models depending of the zone. For example, `GaussianMixture` labels works better at the rear/front view boundary while `KMeans` looks better in total.
<br>

:arrow_up: [to begin](README.md##Skillfactory)

<br><br>
Star ⭐️⭐️⭐️⭐️️⭐️ my project if you like it or think it is useful
