import re
import argparse
from pprint import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out-file', default='', type=str)

args = parser.parse_args()

out_file = open(args.out_file, 'a')

root = 'md_d_metric/aligned/'

f = open(root + "ref_human_detail",'r', encoding='utf-8')
dic={}
insert = 0 
delete = 0
sub = 0
cor=0
count=0
##  0： ref  1：human 2：ops --- 3: human  4： our  5: ops 
for line in f:
    line = line.strip()
    if("ref" in line ):
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +"," ",ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]] = []
        dic[ref[0]].append(ref[1])
    elif( "hyp" in line ):
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +"," ",hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif( " op " in line ):   
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +"," ",op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])
        for i in op_seq:
            if(i == "I"):
                insert+=1
            elif(i == "D"):
                delete+=1
                count+=1
            elif(i == "S"):
                sub +=1
                count+=1
            elif(i=="C"):
                cor+=1
                count+=1
f.close()
# ## 发音错误统计
# print("insert:" ,insert)
# print("delete:" ,delete)
# print("sub:" ,sub)
# print("cor:" ,cor)
# print("sum", count)

f = open(root + "human_our_detail",'r', encoding='utf-8')
for line in f:
    line = line.strip()
    fn = line.split(" ")[0]
    if(fn not in dic):
        continue
    if("ref" in line ):
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +"," ",ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]].append(ref[1])
    elif( "hyp" in line ):
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +"," ",hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif( " op " in line ):
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +"," ",op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])
f.close()

f = open(root + "ref_our_detail",'r', encoding='utf-8')
for line in f:
    line = line.strip()
    fn = line.split(" ")[0]
    if(fn not in dic):
        continue
    if("ref" in line ):
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +"," ",ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]].append(ref[1])
    elif( "hyp" in line ):
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +"," ",hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif( " op " in line ):
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +"," ",op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])
f.close()

cor_cor = 0
cor_cor1 = 0 
cor_nocor = 0

sub_sub = 0
sub_sub1 = 0
sub_nosub = 0

ins_ins = 0
ins_ins1 = 0
ins_noins =0

del_del = 0
del_del1 = 0
del_nodel =0

y_true = []
y_pred = []
yy_true = []
yy_pred = []
# print(hyp)

for i in dic:
    arr = dic[i]
    # del detection 
    TTTTTRRRRR = 0
    wav_id = i
    # print(arr)
    ref_seq = arr[0].split(" ")
    ref_seq3 = arr[6].split(" ")
    our_seq3 = arr[7].split(" ")
    op =  arr[2].split(" ")
    op3 = arr[8].split(" ")
    
    # break
    flag = 0
    for i in range( len(ref_seq) ):
        if(ref_seq[i] == "<eps>"):
            continue
        while(flag < len(ref_seq3) and ref_seq3[flag] == "<eps>"):
            flag+=1  
        if flag < len(ref_seq3) and ( ref_seq[i]  == ref_seq3[flag] and ref_seq[i]!="<eps>" ):
            if( op[i] == "D"  and op3[flag] == "D" ):
                del_del+=1
                yy_true.append(ref_seq[i])
                yy_pred.append('eer')
            elif( op[i] == "D" and op3[flag] != "D" and op3[flag] != "C"):
                del_del1+=1
                pho = ref_seq[i]
                yy_true.append(ref_seq[i])
                yy_pred.append(our_seq3[flag])
                debug = 1
            elif( op[i] == "D" and op3[flag] != "D" and op3[flag] == "C"):
                del_nodel+=1
                pho = ref_seq[i]
                yy_true.append(ref_seq[i])
                yy_pred.append(ref_seq[i])
                debug = 1
            flag+=1  
            

    ## cor ins sub detection 
    ref_seq = arr[0].split(" ")
    human_seq = arr[1].split(" ")
    op =  arr[2].split(" ")
    human_seq2 = arr[3].split(" ")
    our_seq2 = arr[4].split(" ")
    op2 = arr[5].split(" ")

    flag = 0 
    for i in range( len(human_seq) ):
        if(human_seq[i] == "<eps>"):
            continue
        while(human_seq2[flag] == "<eps>"):
            flag+=1
        if( human_seq[i]  == human_seq2[flag] and human_seq[i]!="<eps>" ):
            if( op[i] == "C"  and op2[flag] == "C" ):
                cor_cor+=1
                y_true.append(human_seq[i])
                y_pred.append(human_seq[i])
            elif( op[i] == "C" and op2[flag] != "C"):
                cor_nocor+=1
                y_true.append(human_seq[i])
                y_pred.append(our_seq2[flag])


            if( op[i] == "S" and op2[flag] == "C" ):
                sub_sub+=1
                pho = ref_seq[i]
                yy_true.append(ref_seq[i])
                yy_pred.append(human_seq2[flag])
                debug = 1
            elif( op[i] == "S"  and op2[flag] !="C" and ref_seq[i] != our_seq2[flag]):
                sub_sub1+=1
                pho = ref_seq[i]
                yy_true.append(ref_seq[i])
                yy_pred.append(our_seq2[flag])
                debug = 1

            elif( op[i] == "S"  and op2[flag] !="C" and ref_seq[i] == our_seq2[flag]):
                sub_nosub+=1
                pho = ref_seq[i]
                yy_true.append(ref_seq[i])
                yy_pred.append(ref_seq[i])
                debug = 1
            
            if(op[i] == "I" and op2[flag] == "C" ):
                ins_ins+=1

            elif( op[i] == "I" and op2[flag]!="C" and op2[flag]!="D"):
                ins_ins1+=1
            elif( op[i] == "I" and op2[flag]!="C" and op2[flag]=="D"):
                ins_noins+=1

            flag+=1

sum1 = cor_cor + cor_nocor + sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel  
# print("sum:",sum1)
TR = sub_sub + sub_sub1  +  +del_del1+del_del + ins_ins1 +  ins_ins
FR = cor_nocor # 
FA = sub_nosub + ins_noins + del_nodel 
TA = cor_cor # 
recall = TR/(TR+FA)
# print(TR+FA)
precision = TR/(TR+FR)
print("Recall: %.4f" %(recall), file=out_file)
print("Precision: %.4f" %(precision), file=out_file)
print("f1-score: %.4f" % ( 2*precision*recall/(recall+precision)), file=out_file)

print("True Acception: %.4f | %d" %(cor_cor/(cor_cor+cor_nocor), TA), file=out_file)
print("False Rejection: %.4f | %d" %(cor_nocor/(cor_cor+cor_nocor), FR), file=out_file)
err_count = sub_sub+sub_sub1+sub_nosub+ins_ins+ins_ins1+ins_noins+del_del+del_del1+del_nodel
false_accept = sub_nosub + ins_noins + del_nodel
Correct_Diag = sub_sub + ins_ins + del_del
Error_Diag =  sub_sub1 + ins_ins1 + del_del1
print("False Acceptance: %.4f | %d" %(false_accept/err_count, false_accept), file=out_file)
print("Correct Diagnosis: %.4f | %d" %(Correct_Diag/(Correct_Diag+Error_Diag), Correct_Diag), file=out_file)
print("Error Diagnosis: %.4f | %d" %(Error_Diag/(Correct_Diag+Error_Diag), Error_Diag), file=out_file)
FAR = 1-recall
FRR = cor_nocor/(cor_nocor+cor_cor)
DER = Error_Diag / (Error_Diag + Correct_Diag)
print("False Acceptance Rate: %.4f" %(FAR), file=out_file)
print("False Rejection Rate: %.4f" %(FRR), file=out_file)
print("Diagnosis Error Rate: %.4f" %(DER), file=out_file)
print("Detection Accuracy: %.4f" % ((TA+TR)/(TR+TA+FR+FA)), file=out_file)
# print("sub_sub", sub_sub)
print('\n', file=out_file)