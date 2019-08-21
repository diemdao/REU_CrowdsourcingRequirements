# REU_CrowdsourcingRequirements
Goal: Develop tools and techniques to extract novel ideas and requirements from crowdsourced data. 

Advisor: Dr. Pradeep Murukannaiah

This work is supported by NSF CNS-1757680. 


Background: 
When researchers and businesses crowdsource data from the public, a large database of information can be overwhelming and redundant.
Finding novel ideas from a large data set can be challenging. 

Hence, we developed a study that:
Can automatically determine novel ideas from crowdsourced requirements.
Collected approximately 3,000 requirements from a MTurk survey on what people wanted in their smart home applications.
Filtered the mundane ideas from novel suggestions (requirements).


Current Progress:
Carried out data pre-processing techniques using NLTK (Tokenization, Stop word removal, POS Tagging, Stemming from textual requirements obtained through crowdsourcing). 
Used data mining techniques such as TF-IDF Vectorization and Count Vectorization to understand and analyze the data. 
Implemented K means algorithm and used the Elbow method with Euclidean distance to determine the optimal number of clusters. 
Find the requirements in each cluster.


Direction:
Implement MINAS: Multiclass learning algorithm for novelty detection in data streams.
Create a decision model by training and testing the data set to detect novelty patterns. 
Remove outliers or noise and find concept drifting.
