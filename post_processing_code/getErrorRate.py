## Get error rate.
import pandas as pd
import os
import sys

list_files_names = []
list_y_true = []
list_y_pred = []
list_error_rate = []
list_ASA_error_rate = []
dice_coef_list = []
TP_list = []
FN_list = []
FP_list = []
precision_list = []
recall_list = []
f1_score_list = []
errorRateBasedOnConfusionMatrix = []
accuracy = []

path2Experiment = '../test_exp'
pathToManual = r'../data/Annotation_disector_countFiles/countFiles'


foldNames = ['Iteration_1','Iteration_2','Iteration_3','Iteration_4','Iteration_5','Iteration_6','Iteration_7']
for foldName in foldNames:
    #MiceNames = foldName.split('_')[1:]
    MiceNames = [foldName]
    for mouseName in MiceNames:
        mouseName = mouseName+'_'
        pathToPredictions1 = os.path.join(path2Experiment,foldName,'PredictedMasks2','predMasks','OldPostprocessing_0.5')
        pathToResults = path2Experiment
        pathToPredictions = pathToPredictions1
        def getDiceCoef(diceCoef):
            dice_mean = 0.0
            zeros = 0
            #print(diceCoef['dice'].values)
            for item in diceCoef['dice'].values:
                if item == 0:
                    zeros = zeros + 1
                    continue
                else:
                    dice_mean += item
            dice_mean = dice_mean / (diceCoef.shape[0] - zeros)
            return dice_mean
        def processConfMatrix(df):
            df.columns =['Name','TP','FN','FP']
            df['TP'] = [int(item.split(',')[0].split('  ')[1]) for item in df['TP'].tolist()]
            df['FN'] = [int(item.split(',')[0].split(' ')[2]) for item in df['FN'].tolist()]
            df['FP'] = [int(item.split('  ')[2]) for item in df['FP'].tolist()]
            return df
        folderName = 'Error_analysis_final4'
        if not os.path.isdir(os.path.join(pathToPredictions1,folderName)):
            os.mkdir(os.path.join(pathToPredictions1,folderName))

        for file in os.listdir(pathToPredictions):
            print(file)
            if file.endswith('_count.txt'):
                print('Now serving {}'.format(file))
                fileparts = file.split('_count')
                diceCoef = pd.read_csv(os.path.join(pathToPredictions,fileparts[0]+'_diceCoef.txt'),sep="\t",header=None)
                confMatrix = pd.read_csv(os.path.join(pathToPredictions,fileparts[0]+'_confusionMatrix.txt'),sep="\t",header=None)
                Predicted = pd.read_csv(os.path.join(pathToPredictions,file), sep="\t", header=None)
                #print('dice shape is {}  confMatrix shape is {}  Predicted shape is {}'.format(diceCoef.shape,confMatrix.shape,Predicted.shape))
                Predicted.columns = ['Name', 'Count_predicted']
                diceCoef.columns = ['Name', 'dice']

                limit = Predicted.shape[0]
                print('limit is {}'.format(limit))
                loc = limit   # for lines that has total info
                #dice = diceCoef.iloc[loc,:][1]
                dice_mean = getDiceCoef(diceCoef)
                df_confusion = processConfMatrix(confMatrix)
                TP = df_confusion['TP'].sum()
                FN = df_confusion['FN'].sum()
                FP = df_confusion['FP'].sum()

                Manual = pd.read_csv(os.path.join(pathToManual,'FullDataSetNeuNSingleStainManualAnnotationCount_v2_correctFinalSep6th2018.csv'),sep=",") #FullDataSetNeuNSingleStainManualAnnotationCount_v2_Sep6th2018
                ASA = pd.read_csv(os.path.join(pathToManual,'ASA_count_final_Sep_4_2018.csv'),sep=",")

                print('Predicted shape is {}'.format(Predicted.shape))



                Predicted = Predicted.iloc[0:limit,:]
                diceCoef = diceCoef.iloc[0:limit,:]

                newNames = []
                for item in Predicted['Name'].values:
                    itemParts = item.split('_pred')
                    newNames.append(itemParts[0])
                Predicted['Name']= newNames
                #dice_mean = 0.0
                zeros = 0

                Manual_list = Manual['Name'].values
                Predicted_list= Predicted['Name'].values
                #print(Manual_list)
                #print(Predicted_list)
                intersection = [value for value in Manual_list if value in Predicted_list]
                #print('length of intersection is {}'.format(len(intersection)))
                ## Joining two dataframes.
                #Manual.join(Predicted,on=['Name'],how='inner')
                #result = pd.merge(Manual, Predicted, on='Name',how='inner')
                result = pd.merge(Manual, Predicted, on ='Name', how = 'inner')
                ASA_result = pd.merge(ASA,Predicted,on ='Name', how = 'inner')

                #print(result.head())
                #print(ASA_result.head())
                fractionater  = pd.DataFrame()
                Manual_Unet_ASA = pd.merge(result,ASA_result, on='Name', how='inner')

                #print(Manual_Unet_ASA.head())
                Manual_Unet_ASA['LU'] = Manual_Unet_ASA['Name'].str.split('_').str[0]
                Manual_Unet_ASA['Section'] = Manual_Unet_ASA['Name'].str.split('_').str[1]
                Manual_Unet_ASA['Stack'] = Manual_Unet_ASA['Name'].str.split('_').str[2]
                fractionater['LU'] = Manual_Unet_ASA['LU']
                fractionater['Section'] = Manual_Unet_ASA['Section']
                fractionater['Manual_count'] = Manual_Unet_ASA['Count_x']
                fractionater['Unet_count'] = Manual_Unet_ASA['Count_predicted_x']
                fractionater['ASA_count'] = Manual_Unet_ASA['Count_y']

                sf = fractionater.groupby('LU')['Unet_count'].sum()
                sf1 = fractionater.groupby('LU')['Manual_count'].sum()
                sf2 = fractionater.groupby('LU')['ASA_count'].sum()
                fractionater_total = pd.DataFrame({'LU':sf.index, 'Unet_count':sf.values,'Manual_count':sf1.values,'ASA_count':sf2.values})
                #print(fractionater_total.shape)
                #print(fractionater_total.head())
                fractionater_total.to_csv(os.path.join(pathToPredictions1,folderName,fileparts[0]+'_PerCaseCount_newCount.csv'),sep=',',index=False)

                result.to_csv(os.path.join(pathToPredictions1,folderName,fileparts[0]+'_oldCount.csv'),index=False)
                #result['LU'] = Manual_Unet_ASA['Name'].str.split('_').str[0]
                #result['Section'] = Manual_Unet_ASA['Name'].str.split('_').str[1]
                sf3 = fractionater.groupby(['LU','Section'])['Manual_count'].sum()
                sf4 = fractionater.groupby(['LU','Section'])['Unet_count'].sum()
                sf5 = fractionater.groupby(['LU','Section'])['ASA_count'].sum()
                index = [i[0]+'_'+i[1] for i in sf3.index]
                print(index)
                Count_per_section = pd.DataFrame({'LU_Section':index, 'Manual_count':sf3.values,'Unet_count':sf4.values,'ASA_count':sf5.values})

                #print(result.head())
                #print(Count_per_section.head())
                Count_per_section[['LU_Section','Manual_count','Unet_count','ASA_count']].to_csv(os.path.join(pathToPredictions1,folderName,fileparts[0]+'_PerSectionCount_newCount.csv'),sep=',',index=False)
                #Count_per_section[['LU_Section','Manual_count','Unet_count','ASA_count']].to_csv(os.path.join(pathToPredictions1,'_PerSectionCount.csv'),sep=',',index=False)
                #print(result)
                count_true = result['Count'].values
                count_pred = result['Count_predicted'].values
                count_ASA = ASA_result['Count'].values
                total_true = result['Count'].sum()
                total_pred = result['Count_predicted'].sum()
                error_rate = (abs(total_true - total_pred)/float(total_true))*100
                #print(ASA_result.head())
                #sys.exit()
                ASA_error_rate = (abs(total_true - ASA_result['Count'].sum())/float(total_true))*100
                #print('total_true {} , SUM {} error rate {}'.format(total_true,ASA_result['Count'].sum(),ASA_error_rate))
                #sys.exit()
                TP = int(TP)
                FP = int(FP)
                FN = int(FN)

                FN = abs((TP+FN) - total_true) + FN
                #assert(FN+TP == total_true,'Error in FN count\n')
                #print('new FN is {}'.format(FN))
                precision = TP/ float(TP + FP)
                recall = TP / float(TP + FN)
                #print('temperary remember to remove')
                #precision = 1
                #recall = 1
                f1_score = 2 *precision*recall/float(precision+recall)
                print('Dice coef is {}'.format(dice_mean))
                print('total_true is {}'.format(total_true))
                print('total_pred is {}'.format(total_pred))
                print('total ASA count is {}'.format(ASA_result['Count'].sum()))
                print('Error rate is {}'.format(error_rate))
                print('precision is {}'.format(precision))
                print('recall is {}'.format(recall))
                print('f1_score is {}'.format(f1_score))
                print('ASA error rate {}'.format(ASA_error_rate))
                print('TP {}'.format(TP))
                print('FP {}'.format(FP))
                print('FN {}'.format(FN))
                print('Error new based on confusion Matrix {}'.format((FP+FN)/float(total_pred)))
                print('Accuracy {}'.format(1- ((FP+FN)/float(total_pred))))
                #sys.exit()
                print('--'*30)
                list_files_names.append(mouseName)
                list_y_true.append(total_true)
                list_y_pred.append(total_pred)
                list_error_rate.append(error_rate)
                list_ASA_error_rate.append(ASA_error_rate)
                dice_coef_list.append(dice_mean)
                TP_list.append(TP)
                FN_list.append(FN)
                FP_list.append(FP)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_score_list.append(f1_score)
final = pd.DataFrame()            
final['fold'] = list_files_names
final['y_true'] = list_y_true
final['y_pred'] = list_y_pred
final['error_rate(%)'] = list_error_rate
final['ASA_error_rate(%)'] = list_ASA_error_rate
final['diceCoef'] = dice_coef_list
final['TP'] = TP_list
final['FN'] = FN_list
final['FP'] = FP_list
final['precision'] = precision_list
final['recall'] = recall_list
final['f1_score'] = f1_score_list
final2 = pd.DataFrame()

final2['fold']= ['Avg/Sum']
final2['y_true'] = [final['y_true'].sum()]
final2['y_pred'] = [final['y_pred'].sum()]
final2['error_rate(%)'] = [final['error_rate(%)'].mean()]
final2['ASA_error_rate(%)'] = [final['ASA_error_rate(%)'].mean()]
final2['diceCoef'] = [final['diceCoef'].mean()]
final2['TP'] = [final['TP'].sum()]
final2['FN'] = [final['FN'].sum()]
final2['FP'] = [final['FP'].sum()]
final2['precision'] = [final['precision'].mean()]
final2['recall'] = [final['recall'].mean()]
final2['f1_score'] = [final['f1_score'].mean()]

#parentFolder = os.path.dirname(pathToPredictions1)
if not os.path.exists(os.path.join(pathToResults,"Error_analysis")):
    os.makedirs(os.path.join(pathToResults,"Error_analysis"))
#print(pathToResults)
final.to_csv(os.path.join(pathToResults,"Error_analysis",'error.csv'),sep=',',index=False)
final2.to_csv(os.path.join(pathToResults,"Error_analysis",'errorAvg.csv'),sep=',',index=False)
print('##'*30)
print('Average DL error of folds is {}'.format(final['error_rate(%)'].mean()))
print('True count is {}'.format(final['y_true'].sum()))
print('DL predicted count is {}'.format(final['y_pred'].sum()))
print('DL Std of folds is {}'.format(final['error_rate(%)'].std()))
print('Average ASA error is {}'.format(final['ASA_error_rate(%)'].mean()))
print('ASA Std of folds is {}'.format(final['ASA_error_rate(%)'].std()))
print('Avg DL dice Coef is {}'.format(final['diceCoef'].mean()))
print('DL TP {}'.format(final['TP'].sum()))
print('DL FP {}'.format(final['FP'].sum()))
print('DL FN {}'.format(final['FN'].sum()))
print('Avg DL precision is {}'.format(final['precision'].mean()))
print('Avg DL recall is {}'.format(final['recall'].mean()))
print('Avg DL f1 score is {}'.format(final['f1_score'].mean()))
