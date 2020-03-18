# Semantic Comparison of State-of-the-Art Deep Learning APIs for Image Multi-Label Classification
Some functionallity and metrics calculation implemented for the "[Semantic Comparison of State-of-the-Art Deep Learning APIs for Image Multi-Label Classification](https://arxiv.org/abs/1903.09190)" paper.

## Installation
To run the metrics and functionality, you need the following:
 1. Create a MySQL database named "infomedia".
 2. Import the CSVs in the "Data" folder to the "infomedia" DB.
 3. For metrics calc, download the word2vec model from [here](https://code.google.com/archive/p/word2vec/) and put it in the root folder.
 4. For image inference, download the test images from [here](https://drive.google.com/file/d/1F7Uts0k2p9S7GZqTVaKdcjgw6ZoEauAV/view?usp=sharing)  and extract them to the "first1000images" folder.
 5. You need Python version 3.6+ and PHP version PHP. For the MySQL and PHP installation we recommend the  [WAMP package](http://www.wampserver.com/en/).

## Functionality
Once you installed the pre-requisites, you can run some of the functionalities. The results for each calculation will be saved in the "results" folder.
 1. To calculate the label-based metrics. Run the PHP script (usually from the PHP installed folder)
     > $php file_path\label_based_metrics.php
 2. To calculate the semantic and example-based metrics, count the "unknown" labels or calc the labels frequencies, run the following Python script. 
     >$python example_based_metrics.py

## Reference

	@article{Kubany2019SemanticClassification, 
	title = {{Semantic Comparison of State-of-the-Art Deep Learning Methods for Image Multi-Label Classification}}, 
	year = {2019}, 
	journal = {arXiv preprint arXiv: 1903.09190}, 
	author = {Kubany, Adam and Ben Ishay, Shimon and Ohayon, Ruben-Sacha and Shmilovici, Armin and Rokach, Lior and Doitshman, Tomer} 
	}
For questions please contact Adam Kubany via [email](adamku@post.bg.ac.il).
