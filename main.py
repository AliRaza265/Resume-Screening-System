from cv_xtractor.extract_entities import extract_entities_from_file
import json


cv_path  =  "data_set/Ryan_Xia_CV_Data_AI.docx"

try :
    print("PLz Wait : ")
    data_extrat = extract_entities_from_file(cv_path)
    print(json.dumps(data_extrat,indent = 4))


except Exception as e:
     print(e) 