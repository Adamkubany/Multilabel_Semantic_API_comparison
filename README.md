# Comparison of State-of-the-Art Deep Learning APIs for Image Multi-Label Classification using Semantic Metrics
Some functionality and metrics calculation implementation for the "[Comparison of State-of-the-Art Deep Learning APIs for Image Multi-Label Classification using Semantic Metrics](https://arxiv.org/abs/1903.09190)" paper.

## Functionality

This repository includes the following functionality:
-   ETL procedure for the ‘Visual Genome’ and ‘Open Images’ datasets
-   Multi-labels classification inference for the paper applied commercial and open-source APIs
-   Evaluation metrics:
	-   Example-based metrics
	-   Semantic metrics
	-   Label-based metrics

 
## ETL Procedure
The ETL procedure for the 'Visual Genome' and 'Open Images' benchmark datasets.
#### Installation:
Download the dataset's metadata to the appropriate folder (\datasets\\[DATASET]\metadata). The [objects.json](https://visualgenome.org/static/data/dataset/objects_v1.json.zip) file for the Visual Genome dataset, and the [train-annotations-human-imagelabels-boxable.csv](https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv) and the [train-images-boxable-with-rotation.csv](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv) for the Open Images dataset.
#### Run command:
     $python etl.py
The ETL results are saved in the  `\datasets\[DATASET]\data` folder. 

## Multi-labels classification inference
Inference scripts for the paper applied commercial and open-source APIs.

### Commercial APIs

The inference script for the following  commercial APIs:
 - Imagga
 - IBM Watson
 - Clarifai
 - Microsoft Computer Vision
 - Google Cloud Vision
 - Wolfram Alpha
#### Installation:
1. Create an inference python conda environment using [`env_inference.yml`](https://github.com/Adamkubany/Multilabel_Semantic_API_comparison/blob/master/sources/env_inference.yml "env_inference.yml") file.
2. Subscribe to each commercial API (following the explanation links inside the APIs' function docstring comments) and set the appropriate API subscription keys in the `AUTH` variable in the [`config.py`](https://github.com/Adamkubany/Multilabel_Semantic_API_comparison/blob/master/config.py "config.py").
#### Run command:
     $python labels_inference_commercial.py
Please note that the commercial APIs often change their interface...
The inference results are saved in the  `\datasets\[DATASET]\data` folder. 


### Open-source APIs
The inference script for the following open-source APIs:

 - InceptionResNet v2 (trained on ImageNet)
 - Mobilenet v2 (trained on ImageNet)
 - VGG19 (trained on ImageNet)
 - Inception v3 (trained on ImageNet)
 - ResNet50 (trained on ImageNet)
 - ResNet50 (trained on COCO)
 - YOLO V3 (trained on ImageNet)
 - YOLO V3 (trained on COCO)
 - Deepdetect (trained on ImageNet)
#### Installation:
1. Create an inference python conda environment using [`env_inference.yml`](https://github.com/Adamkubany/Multilabel_Semantic_API_comparison/blob/master/sources/env_inference.yml "env_inference.yml") file (same env as for the commercial APIs).
2. Python will automatically download the needed model files for the first inference usage of each of the ImageNet trained APIs  (besides the YOLO ImageNet API)
3. For the COCO trained APIs, download the [ResNet50](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5) or  [YOLO](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5) model files to the [`sources`](https://github.com/Adamkubany/Multilabel_Semantic_API_comparison/tree/master/sources "sources") folder.
4. The ImageNet trained YOLO API requires Linux OS (others can manage with Windows) and the follow the [installation procedure](https://pjreddie.com/darknet/imagenet/#darknet53_448).

#### Run command:
     $python labels_inference_open_source.py
The inference results are saved in the  `\datasets\[DATASET]\data` folder. 

## Evaluation metrics
Example-based, label-based, and the proposed semantic evaluation metrics.

### Example based and semantic metrics
- Example-based metrics: accuracy, recall, precision, and F1
- Semantic metrics: semantic accuracy, semantic recall, semantic precision, semantic F1, Word Moving Distance (WMD), fine-tuned BERT, fine-tuned RoBERTa
#### Installation:
1. Create a metrics calculation python conda environment using [`env_metrics_calc.yml`](https://github.com/Adamkubany/Multilabel_Semantic_API_comparison/blob/master/sources/env_metrics_calc.yml "env_metrics_calc.yml") file.
2. Download the [word2vec pre-trained bin file](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) to the [`sources`](https://github.com/Adamkubany/Multilabel_Semantic_API_comparison/tree/master/sources "sources") folder.
3. Python will automatically download the needed model files for the first use of BERT and RoBERTa.


#### Run command:
     $python example_based_metrics_semantics.py
The metrics results are saved in the  `\datasets\[DATASET]\results` folder. 

### Label based metircs
Label-based metrics: micro and macro averaging of precision, recall, and F1
#### Installation:
1. Install PHP 7 and MySQL, we recommend the  [WAMP package](http://www.wampserver.com/en/).
2. Create the `sem_open_images` and `sem_visual_genome` MySQL databases and import the DB tables using the `.sql` file in the `\datasets\\[DATASET]\data` folder.
#### Run command:
     $php label_based_metrics.php "OPEN_IMAGE" (or "VISUAL_GENOME")
The metrics results are saved in the  `\datasets\[DATASET]\results` folder. 


For questions and remarks please contact [Adam Kubany](adamku@post.bgu.ac.il).
