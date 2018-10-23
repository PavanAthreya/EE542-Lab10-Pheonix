
##################### YUE SHI CODE STARTS HERE ################################


import requests
import json
import pandas as pd
import os

def retrieveFileMeta(file_ids,outputfile):
    '''

    Get the tsv metadata for the list of case_ids
    Args:
        file_ids: numpy array of file_ids
        outputfile: the output filename

    '''

    fd = open(outputfile,'w')
    cases_endpt = 'https://api.gdc.cancer.gov/files'

    # The 'fields' parameter is passed as a comma-separated string of single names.
    fields = [
        "file_id",
        "file_name",
        "cases.submitter_id",
        "cases.case_id",
        "data_category",
        "data_type",
        "cases.samples.tumor_descriptor",
        "cases.samples.tissue_type",
        "cases.samples.sample_type",
        "cases.samples.submitter_id",
        "cases.samples.sample_id",
        "cases.samples.portions.analytes.aliquots.aliquot_id",
        "cases.samples.portions.analytes.aliquots.submitter_id",
        "cases.project.primary_site",
        "cases.project.disease_type"
        ]

    filters = {
        "op":"in",
        "content":{
            "field":"files.file_id",
            "value": file_ids.tolist()
        }
    }
    #print(filters)
    fields = ','.join(fields)

    params = {
        "filters" : filters,
        "fields": fields,
        "format": "TSV",
        "pretty": "true",
        "size": 12000
    }
    # print (params)
    #print (filters)
    #print (fields)
    
    
    response = requests.post(cases_endpt, headers = {"Content-Type": "application/json"},json = params)
    fd.write(response.content.decode("utf-8"))
    fd.close()

    # print(response.content)
def retrieveCaseMeta(file_ids,outputfile):
    '''

    Get the tsv metadata for the list of case_ids
    Args:
        file_ids: numpy array of file_ids
        outputfile: the output filename

    '''

    fd = open(outputfile,'w')
    cases_endpt = 'https://api.gdc.cancer.gov/cases'


    filters = {
        "op":"in",
        "content":{
            "field":"cases.case_id",
            "value": file_ids.tolist()
        }
    }

    # print (filters)
    #expand group is diagnosis and demoragphic
    params = {
        "filters" : filters,
        "expand" : "diagnoses,demographic,exposures",
        "format": "TSV",
        "pretty": "true",
        "size": 12000
    }
    # print (params)
    #print (filters)
    #print (fields)
    
    
    response = requests.post(cases_endpt, headers = {"Content-Type": "application/json"},json = params)
    # print (response.content.decode("utf-8"))
    fd.write(response.content.decode("utf-8"))
    fd.close()

def genCasePayload(file_ids,payloadfile):
    '''
    Used for the curl method to generate the file payload.
    '''

    fd = open(payloadfile,"w")
    filters = {
        "filters":{
            "op":"in",
            "content":{
                "field":"cases.case_id",
                "value": file_ids.tolist()
            }
        },
        "format":"TSV",
        "expand" : "diagnoses,demographic,exposures",
        "size": "1000",
        "pretty": "true"
    }
    json_str = json.dumps(filters)
    fd.write(json_str)
    fd.close()
    # return json_str

def genFilePayload(file_ids,payloadfile):
    '''
    Used for the curl method to generate the payload.
    '''


    fd = open(payloadfile,"w")
    filters = {
        "filters":{
            "op":"in",
            "content":{
                "field":"files.file_id",
                "value": file_ids.tolist()
            }
        },
        "format":"TSV",
        "fields":"file_id,file_name,cases.submitter_id,cases.case_id,data_category,data_type,cases.samples.tumor_descriptor,cases.samples.tissue_type,cases.samples.sample_type,cases.samples.submitter_id,cases.samples.sample_id,cases.samples.portions.analytes.aliquots.aliquot_id,cases.samples.portions.analytes.aliquots.submitter_id",
        "pretty":"true",
        "size": "1000"
    }
    json_str = json.dumps(filters)
    fd.write(json_str)
    fd.close()




def curlFileMeta(file_ids,payloadfile,outputfile):
    genFilePayload(file_ids,payloadfile)
    os.system("curl --request POST --header \"Content-Type: application/json\" --data @"+payloadfile+" 'https://api.gdc.cancer.gov/files' > "+outputfile)

def curlCaseMeta(case_ids,payloadfile,outputfile):
    genCasePayload(case_ids,payloadfile)
    os.system("curl --request POST --header \"Content-Type: application/json\" --data @"+payloadfile+" 'https://api.gdc.cancer.gov/cases' > "+outputfile)





if __name__ == '__main__':

    data_dir = "/Users/vishadnehal/Desktop/EE542/Lab10/vishad/data/"
    filename = data_dir+"file_case_id_DNA.csv"
    
    
    df = pd.read_csv(filename)
    file_ids = df.file_id.values
    case_ids = df.case_id.values
    # print(case_ids)
    
    fileids_meta_outfile = data_dir + "files_meta.tsv"
    caseids_meta_outfile = data_dir + "cases_meta.tsv"
    # python request method
    retrieveFileMeta(file_ids,fileids_meta_outfile)
    retrieveCaseMeta(case_ids,caseids_meta_outfile)



    # the curl method
    '''
    filepayload = "FilePayload"
    casepayload = "CasePayload"
    fileids_meta_outfile = "curl_files_meta.tsv"
    caseids_meta_outfile = "curl_cases_meta.tsv"
    curlFileMeta(file_ids,filepayload,fileids_meta_outfile)
    curlCaseMeta(case_ids,casepayload,caseids_meta_outfile)
    '''
    
