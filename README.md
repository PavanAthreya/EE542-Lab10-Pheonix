EE542 - Internet and Cloud Computing
Fall 2018 - Lab10 - Genomics Common Data
Professor Young H. Cho

Team Assignment

Team Members:
A - Ekta Trivedi (etrivedi@usc.edu)
B - Vishad Shah (vishadsh@usc.edu)
C - Pavan Athreya (pavan.athreya@usc.edu)

Youtube Video Link: https://www.youtube.com/watch?v=4maCzzg9zDw&t=1s

Genomics Common Data
	1	Run the check.py file. This file should be in the same folder as where the miRNA.csv is present. All files are compatible with Python 3 and above versions eg : $python3 check.py
	2	Run the parse_file_case_id.py script to get the unique file id for corresponding case ids. After this get the JSON file from the genomimcs data repository.
	3	Run the request_meta.py to get the meta data for all the cases
	4	Run the gen_miRNA_matrix.py to get the miRNA matrix and labels for all the data
	5	Run the test.py file for running the machine learning algorithm which is Linear Regression. The precision , accuracy , F1-score and sensitivity values can be seen in the graph. The scatterplots are also plotted for showing the variation per principal component
