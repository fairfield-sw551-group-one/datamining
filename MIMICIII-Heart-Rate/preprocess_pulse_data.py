import pandas as pd

inputfile = open("CHARTEVENTS.csv", 'r')
outputfile = open("CHARTEVENTS_HR.csv", 'w')

#IDs for pulse, with erroneous values removed
#Obtained using `grep -i 'pulse' D_ITEMS.csv`
#These are string descriptors
#pulselist = [546,549,603,1438,1456,81,156,188,1332,1341,3124,1725,1740,1909,8144,1505,2530,2702,3568,6347,7090,8489,8372,8381,8386,8455,8456,8473,223944,223945,223946,223947,223948,223949,228194,223934,223935,223936,223938,223939,223940,223941,223943]
#grep -i 'Heart Rate' D_ITEMS.csv
#These are numeric descriptors
pulselist = [220045]
count = 0

for line in inputfile:
	linecolumns = line.split(",")
	if count == 0:
		#print(line)
		outputfile.write(line)
	elif int(linecolumns[4]) in pulselist:
		#print("Found "+linecolumns[4]+" ID on line")
		#print(line)
		outputfile.write(line)
	count += 1

inputfile.close()
outputfile.close()

df = pd.read_csv("CHARTEVENTS_HR.csv")
#"ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","STORETIME","CGID","VALUE","VALUENUM","VALUEUOM","WARNING","ERROR","RESULTSTATUS","STOPPED"
cols = ['ROW_ID','ITEMID','CGID','STORETIME','VALUE','VALUEUOM','WARNING','ERROR','RESULTSTATUS','STOPPED']
df.drop(cols, axis=1, inplace=True)
df = df.sort_values(['SUBJECT_ID','CHARTTIME'])
df.rename(columns={'VALUENUM':'HEART_RATE'}, inplace=True)
df.to_csv("CHARTEVENTS_HR_FILTERED.csv")

