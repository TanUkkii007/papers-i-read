'''
A format script for The SIWIS French Speech Synthesis Database.
http://datashare.is.ed.ac.uk/handle/10283/2353
'''

import os
import csv

part = "part1"
text_dir = "text/" + part + "/"
output_filename = "text.csv"

files = os.listdir(text_dir)

with open(output_filename, "w") as output:
    writer = csv.writer(output)
    for filename in files:
        with open(text_dir + filename, "r") as f:
            data = f.read()
            writer.writerow([part + "/" + filename.replace(".txt", ""), data])
