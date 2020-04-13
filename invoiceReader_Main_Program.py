from tesserocr import PyTessBaseAPI, RIL, iterate_level,PSM
import pandas as pd
import numpy as np
import os,sys,copy,re,json
from sklearn.externals import joblib
NB_model = open('NB_model.pkl','rb')
mnb = joblib.load(NB_model)
NB_vect = open('NB_vect.pkl','rb')
vect = joblib.load(NB_vect)
output_df = pd.DataFrame(columns=['Filename','Extracted Values'])
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def predict(sent):
    sent = pd.Series([sent])
    sent_transformed = vect.transform(sent)
    ans = mnb.predict(sent_transformed)[0]
    return ans

def extract_data_from_image(filename): 
    print(filename)
    bboxPrev = None
    coldist = None
    pdfPageDf = pd.DataFrame(columns=['trueline','lineitem','bbox','coldist'])
    with PyTessBaseAPI(path = "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata",psm=PSM.SPARSE_TEXT_OSD) as api:
        api.SetImageFile(filename)
        api.Recognize()
        ri = api.GetIterator()      
        level = RIL.TEXTLINE
        for r in iterate_level(ri, level):        
            pdfLine = r.GetUTF8Text(level)
            trueline = True
            bbox = r.BoundingBoxInternal(level)
            
            if not bboxPrev == None:
                if abs(bbox[1]-bboxPrev[1]) <=15:
                    trueline = False
                    coldist = dist(np.array([bbox[0],bbox[1]]),np.array([bboxPrev[0],bboxPrev[1]]))
            bboxPrev = bbox        
            pdfPageDf = pdfPageDf.append({"trueline":trueline,"lineitem":pdfLine,"bbox": bbox ,"coldist":coldist},ignore_index=True)
            # print(pdfLine.strip(),bbox)
    return pdfPageDf
    
def optimizeLine(pageDf):
    # normDf = pd.DataFrame(columns=['lineitem','bbox'])
    # countLine = 0 
    # print("df length ",(len(pageDf) -1) )
    for i in range(0,(len(pageDf) -1)):
        # print("i value ->",i)
        count = copy.deepcopy(i)
        # print(i,count,pageDf.iloc[count][0],pageDf.iloc[count+1][0])
        lineitem = str(pageDf.iloc[count][1]).strip()
        
        
        while pageDf.iloc[count +1][0] == False:
            # print("inside while")
            lineitem = lineitem +" "+ str(pageDf.iloc[count+1][1]).strip()
            
            #
            count +=1
            if count >= len(pageDf)-1:
                break
        # print(i,count)
        # print(lineitem)
        count = 0
        pageDf.at[i,'lineitem'] = lineitem
        
    return pageDf
    
    

def findLocation(index,value):
    # print(index,value)
    if value in df_master.iloc[index][2]:
        for j in range(index+1,len(df_master)-1):
            # print(df_master.iloc[j][2])
            if len(df_master.iloc[j][2].strip()) > 0 and  value.strip() in df_master.iloc[j][2].split()[0]:
                return df_master.iloc[j][3]
            elif df_master.iloc[j][1] == True:
                break
        return df_master.iloc[index][3]
    else:
        return None
    
def findreqamount(index,nums,line = ""):
    locs = {}
    count_locs = 0
    
    for n in nums:
        line = re.sub( re.escape(n),"",line )
        n_loc = findLocation(index,n)
        if n_loc == None:
            continue
        # n_dist = int(dist( np.array([n_loc[0],n_loc[1]]),np.array([ req_yr_location[0],req_yr_location[1] ] ) ))
        # locs[count_locs] = n_dist
        # count_locs +=1    
    # print(locs)
        if line.strip().endswith(")"):
            line = line[:-1]
        if req_yr_location != None and abs(n_loc[0] - req_yr_location[0]) < 100:
            return line,n
            
    return line,"NaN"
def clean_df(row):
    row = re.sub("(?i)notes","",row)
    # row = re.sub("\(","-",row)
    # row = re.sub("\)","",row)
    row = re.sub(",\s?","",row)
    row = re.sub("(?i)^A.*f$","",row)
    return row     
for root, dirs, files in os.walk(os.path.join(os.getcwd(),"data")):
    path = root.split(os.sep)
    # print((len(path) - 1) * '---', os.path.basename(root))
    
    for file in files:
        output_dict = {}
        if file.endswith(".jpg"):
            
            output_list = []
            df = extract_data_from_image(os.path.join(root,file))
            ### print(df)
            df_master = optimizeLine(df)
            df_master["lineitem"] = df_master["lineitem"].apply(clean_df) 
            df_master.reset_index(inplace=True)
            df_trueline = df_master[df_master['trueline']]
            # print(df_master) 
            req_yr_location = None
            yr_list = ["2016","2017","2018","2019","2020"]
            for k in range(0,len(df_trueline)):
                if str(df_trueline.iloc[k][2]).strip() != "":
                    if len(df_trueline.iloc[k][2].split()) <= 3 and  any( yr in  df_trueline.iloc[k][2] for yr in yr_list):
                        if "2019" in df_trueline.iloc[k][2]:
                            req_yr_location = findLocation(df_trueline.iloc[k][0],"2019")
                            print("2019 location => ",req_yr_location)
                if predict(df_trueline.iloc[k][2]) == 2:
                    # print(df_trueline.iloc[k][2])
                    numbers = re.findall(r"\(?\d+\)?",df_trueline.iloc[k][2])
                    a,b = findreqamount(df_trueline.iloc[k][0],numbers,df_trueline.iloc[k][2])
                    if a != None and len(str(a).strip()) > 0 and not any( yr in  df_trueline.iloc[k][2] for yr in yr_list):
                        if str(b).strip().startswith("("):
                            b = b[1:]
                            b = "-"+b
                        if str(b).strip().endswith(")"):
                            b = b[:-1]
                            
                        output_list.append((a,b))
            # print(output_list)
            output_dict = json.dumps(dict(output_list))
            output_df = output_df.append({"Filename":file[:-8],"Extracted Values":output_dict},ignore_index=True)
        print(output_df)
        print(len(output_df))

output_df.to_csv("output.csv")

