#copyright: yueshi@usc.edu
import pandas as pd 
import json
from multiprocessing import Pool

def processFile(inputfile,outputfile):
	'''
	read the json file and parse the file id and case id info and save it 
	'''
	with open(inputfile) as data_file:    
		data = json.load(data_file)

	data_arr = []
	case_ids = set()
	for each_record in data:
		# print (each_record)
		file_id = each_record['file_id']
		case_id =  each_record['cases'][0]['case_id']
		if case_id in case_ids:
			case_ids.add(case_id)

		else:
			
			data_arr.append([file_id,case_id])

	df = pd.DataFrame(data_arr, columns = ['file_id','case_id'])
	
	df.to_csv(outputfile,index=False)
	

if __name__ == '__main__':


	# modify the input file path when use it.
	data_dir = "/Users/vishadnehal/Desktop/EE542/Lab10/vishad/data/"
	inputFile = data_dir + "files.2018-10-18.json"
	outputFile =  data_dir + 'file_case_id_DNA.csv'
	processFile(inputFile, outputFile)




 




