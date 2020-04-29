# # Comparison of State-of-the-Art Deep Learning APIs for Image Multi-Label Classification using Semantic Metrics
Some functionallity and metrics calculation implemention for the "[Comparison of State-of-the-Art Deep Learning APIs for Image Multi-Label Classification using Semantic Metrics](https://arxiv.org/abs/1903.09190)" paper.

# Installation
To run the metrics and functionality, you need the following steps to prepare the environment:
 1. You need Python 3.7 and PHP 7. For the MySQL and PHP installation we recommend the  [WAMP package](http://www.wampserver.com/en/).
 2. Create a MySQL database named "infomedia".
 3. Import the CSVs in the "Data" folder to the "infomedia" DB.
 4. For metrics calc, download the word2vec model from [here](https://code.google.com/archive/p/word2vec/) and put it in the "WordEmbedding" folder.
 5. For image inference, download the test images from [here](https://drive.google.com/file/d/1F7Uts0k2p9S7GZqTVaKdcjgw6ZoEauAV/view?usp=sharing)  and extract them to the "first1000images" folder.

## Functionality
Once you installed the pre-requisite environment, you can run some of the functionalities. The results for each calculation will be saved in the "Results" folder.
 1. To calculate the label-based metrics. Run the PHP script (usually from the PHP installed folder)
     > $php file_path\label_based_metrics.php
 2. To calculate the semantic and example-based metrics, count the "unknown" labels or calc the labels frequencies, run the following Python script. 
     >$python example_based_metrics.py

## References

	@article{Kubany2019SemanticClassification, 
	title = {{Semantic Comparison of State-of-the-Art Deep Learning Methods for Image Multi-Label Classification}}, 
	year = {2019}, 
	journal = {arXiv preprint arXiv: 1903.09190}, 
	author = {Kubany, Adam and Ben Ishay, Shimon and Ohayon, Ruben-Sacha and Shmilovici, Armin and Rokach, Lior and Doitshman, Tomer} 
	}
For questions please contact Adam Kubany via  [email](https://github.com/Adamkubany/Multilabel_Semantic_API_comparison/blob/master/adamku@post.bg.ac.il).
