# **Code Overview**

This repository includes a complete pipeline for generating the Urban Color Dataset using Google Street View Imagery and Vision-Language Models.



### **1. Image Collection** 
`scripts/01_image_collection.ipynb`


Extracts metadata, and downloads front/back direction images per point across the road network via Google Street View API.


> You need to obtain your own Google Maps API key and enable the [Street View Static API](https://developers.google.com/maps/documentation/streetview/overview).



### **2. Weather Classification** 
`scripts/02_weather_classification.py`


Classifies images into `sunny` or `cloudy` using CLIP.

> CLIP model is not included in this repository. You need to set up [CLIP](https://github.com/openai/CLIP) locally.



### **3. Semantic Segmentation** 
`scripts/03_semantic_segmentation.py`


Performs semantic segmentation using Grounded SAM model to segment images into 34 initial semantic categories.

> Grounded SAM is not included in this repository. You need to deploy [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) separately.



### **4. Color Extraction** 
`scripts/04_01_learn_color_centroids.py`


- Implements incremental online k-means clustering in the RGB color space to extract initial 50 colors.


`scripts/04_02_apply_color_model.py`


- Generates a semantic-color matrix (`data/intermediate/color_semantic_raw.parquet`) that stores pixel ratios for all images across 34 semantic categories and 50 color clusters.



### **5. Dataset Integration** 
`scripts/05_dataset_integration.ipynb`



- Merges:

  - 50 → 22 color classes (CIEDE2000 color difference)
  - 34 → 18 semantic classes
  - Combines all with weather labels

- Produces the final integrated dataset (`data/final/color_semantic_merged_with_weather.parquet`).
- Visualizes semantic-color relationships and supports downstream statistical/spatial analysis.







# **Dataset Description**
The integrated dataset file is located at: `data/final/color_semantic_merged_with_weather.parquet`. It contains 75,852 records, each representing a Google Street View image location sampled from the road network of Hong Kong, including three groups of attributes:


### 1. Metadata

* `longitude`: Longitude of the image location (WGS84)
* `latitude`: Latitude of the image location (WGS84)
* `image_direction`: Camera view direction, either `Front` or `Back`



### 2. Weather Label

* `weather`: Lighting condition of the image, classified as `sunny` or `cloudy`



### 3. Semantic-Color Features


* There are **396** numeric columns, each named using the format:
  `id{i}_center{j}`
  where:

  * `i ∈ [1, 18]`: semantic class ID 
  * `j ∈ [0, 21]`: color class ID 

* Each value represents the **pixel ratio** in the image that belong to semantic class *i* and fall into color class *j*. 

* Class ID mappings are in `data/final/merged_semantic_info.json` and `data/final/merged_color_info_with_hex.json`.

* For each row (image), the 396 values sum to **1.0**.




# **Legal Notice**

This dataset does not contain any Google Street View images, visual renderings, or screenshots. All included data—such as weather classification, semantic-color distributions—are derived features generated through automated processing of imagery accessed via the official Google Street View API.

The dataset strictly avoids redistribution or modification of any original copyrighted content from Google. Users must independently access any imagery through authorized API usage and comply with the [Google Maps Platform Terms of Service](https://cloud.google.com/maps-platform/terms).

> © Google  
> Data obtained via Google Street View API.


