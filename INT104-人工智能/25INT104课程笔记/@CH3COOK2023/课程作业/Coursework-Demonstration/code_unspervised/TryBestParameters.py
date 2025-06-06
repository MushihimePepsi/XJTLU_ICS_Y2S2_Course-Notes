import pandas as pd
import DataProcessing as DP
import Run as R


# 文件路径(需要手动更改)
path = 'training_set.xlsx'
# 对原始数据的 Null值进行处理
rawData = pd.read_excel(path)
rawData = DP.deleteNaNValueRow(rawData)

# KMEANS----------------------------------------------------------------------------------------------------------------------------------------------
def getKMeansStandardizeBest():
    mode = 'standardize'
    finalScore = 20
    combination1 = DP.select_column(rawData,'Programme','Gender','Grade')
    combination2 = DP.select_column(rawData,'Programme','Gender','Q1')
    combination3 = DP.select_column(rawData,'Programme','Gender','Q5')
    finalScore = min(finalScore,R.ratio_kmeans(initMethod = mode,feature_1=combination1,feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore

def getKMeansMinMaxBest():
    mode = 'minMax'
    finalScore = 20
    combination1 = DP.select_column(rawData,'Programme','Gender','Grade')
    combination2 = DP.select_column(rawData,'Programme','Gender','Q1')
    combination3 = DP.select_column(rawData,'Programme','Gender','Q2')
    finalScore = min(finalScore,R.ratio_kmeans(initMethod = mode,feature_1=combination1,feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore

def getKMeansNormalizeBest():
    mode = 'normalize'
    finalScore = 20
    combination1 = DP.select_column(rawData,'Programme','Gender','Grade')
    combination2 = DP.select_column(rawData,'Programme','Gender','Q1')
    combination3 = DP.select_column(rawData,'Programme','Gender','Q2')
    finalScore = min(finalScore,R.ratio_kmeans(initMethod = mode,feature_1=combination1,feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore
# GMM-------------------------------------------------------------------------------------------------------------------------------------------------
def getGMMStandardizeBest():
    mode = 'standardize'
    finalScore = 20
    combination1 = DP.select_column(rawData,'Programme','Gender','Grade')
    combination2 = DP.select_column(rawData,'Programme','Gender','Q1')
    combination3 = DP.select_column(rawData,'Programme','Gender','Q2')
    finalScore = min(finalScore,R.ratio_gmm(initMethod = mode,feature_1=combination1,feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore

def getGMMMinMaxBest():
    mode = 'minMax'
    finalScore = 20
    combination1 = DP.select_column(rawData,'Programme','Gender','Grade')
    combination2 = DP.select_column(rawData,'Programme','Gender','Q1')
    combination3 = DP.select_column(rawData,'Programme','Gender','Q2')
    finalScore = min(finalScore,R.ratio_gmm(initMethod = mode,feature_1=combination1,feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore


def getGMMNormalizeBest():
    mode = 'normalize'
    finalScore = 20
    combination1 = DP.select_column(rawData,'Programme','Gender','Grade')
    combination2 = DP.select_column(rawData,'Programme','Gender','Q1')
    combination3 = DP.select_column(rawData,'Programme','Gender','Q2')
    finalScore = min(finalScore,R.ratio_gmm(initMethod = mode,feature_1=combination1,feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore

# Hierarchical-----------------------------------------------------------------------------------------------------------------------------------
def getHierarchicalStandardizeBest():
    mode = 'standardize'
    finalScore = 20
    combination1 = DP.select_column(rawData, 'Programme', 'Gender', 'Grade')
    combination2 = DP.select_column(rawData, 'Programme', 'Gender', 'Q1')
    combination3 = DP.select_column(rawData, 'Programme', 'Gender', 'Q2')
    finalScore = min(finalScore, R.ratio_hierarchical(initMethod=mode, feature_1=combination1, feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore

def getHierarchicalMinMaxBest():
    mode = 'minMax'
    finalScore = 20
    combination1 = DP.select_column(rawData, 'Programme', 'Gender', 'Grade')
    combination2 = DP.select_column(rawData, 'Programme', 'Gender', 'Q1')
    combination3 = DP.select_column(rawData, 'Programme', 'Gender', 'Q2')
    finalScore = min(finalScore, R.ratio_hierarchical(initMethod=mode, feature_1=combination1, feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore

def getHierarchicalNormalizeBest():
    mode = 'normalize'
    finalScore = 20
    combination1 = DP.select_column(rawData, 'Programme', 'Gender', 'Grade')
    combination2 = DP.select_column(rawData, 'Programme', 'Gender', 'Q1')
    combination3 = DP.select_column(rawData, 'Programme', 'Gender', 'Q2')
    finalScore = min(finalScore, R.ratio_hierarchical(initMethod=mode, feature_1=combination1, feature_2=combination2,feature_3=combination3,dimension=2)[0])
    return finalScore