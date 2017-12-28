#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "the number of people:", len(enron_data)
print type(enron_data)
print enron_data.values()[0]
print "the number of features:", len(enron_data.values()[0].keys())

poi_count = 0

for people in enron_data.values():
    if people['poi']:
        poi_count += 1
print "the number of POI:", poi_count

f = open("../final_project/poi_names.txt")

import re
pattern = r'\w+, \w+'

poi_count_in_txt = 0
for string in f.readlines():
    find_string = re.search(pattern, string)
    if find_string != None:
        print find_string.group()
        poi_count_in_txt += 1

print "total poi in txt:", poi_count_in_txt


poi_NaN_count = 0
for people in enron_data.values():
    if people['total_payments'] == "NaN" and people['poi'] == True:
        poi_NaN_count += 1
print "the NaN in total payments:", poi_NaN_count



