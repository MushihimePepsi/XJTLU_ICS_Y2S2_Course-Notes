
import DataProcessing as DP
import Model as MODEL

def ratio_kmeans(initMethod, feature_1, feature_2, feature_3, dimension, attemptTime=10):
    if initMethod == 'standardize':
    # 计算KMeans_standardize
        feature_1_standardize = DP.standardizeData(feature_1)
        feature_2_standardize = DP.standardizeData(feature_2)
        feature_3_standardize = DP.standardizeData(feature_3)
        feature_mix_standardize = DP.concat_multiple_dfs(feature_1_standardize, feature_2_standardize,
                                                         feature_3_standardize)
        feature_mix_standardize = DP.pcaData(feature_mix_standardize, dimension)
        return MODEL.kmeans_clustering(feature_mix_standardize, attemptTime, 4)
    # 计算KMeans_minMax
    if initMethod == 'minMax':

        feature_1_minMax = DP.minMaxData(feature_1)
        feature_2_minMax = DP.minMaxData(feature_2)
        feature_3_minMax = DP.minMaxData(feature_3)

        feature_mix_minMax = DP.concat_multiple_dfs(feature_1_minMax, feature_2_minMax, feature_3_minMax)

        feature_mix_minMax = DP.pcaData(feature_mix_minMax, dimension)
        return MODEL.kmeans_clustering(feature_mix_minMax, attemptTime, 4)


    # 计算KMeans_normalize
    if initMethod == 'normalize':
        feature_1_normalize = DP.minMaxData(feature_1)
        feature_2_normalize = DP.minMaxData(feature_2)
        feature_3_normalize = DP.minMaxData(feature_3)
        feature_mix_normalize = DP.concat_multiple_dfs(feature_1_normalize, feature_2_normalize, feature_3_normalize)
        feature_mix_normalize = DP.pcaData(feature_mix_normalize, dimension)
        return MODEL.kmeans_clustering(feature_mix_normalize, attemptTime, 4)
    return None

def ratio_gmm(initMethod, feature_1, feature_2, feature_3, dimension, attemptTime=10):
    if initMethod == 'standardize':
        # 计算GMM_standardize
        feature_1_standardize = DP.standardizeData(feature_1)
        feature_2_standardize = DP.standardizeData(feature_2)
        feature_3_standardize = DP.standardizeData(feature_3)
        feature_mix_standardize = DP.concat_multiple_dfs(feature_1_standardize, feature_2_standardize,
                                                         feature_3_standardize)
        feature_mix_standardize = DP.pcaData(feature_mix_standardize, dimension)
        return MODEL.gmm_clustering(feature_mix_standardize, attemptTime, dimension)
    # 计算GMM_minMax
    if initMethod == 'minMax':
        feature_1_minMax = DP.minMaxData(feature_1)
        feature_2_minMax = DP.minMaxData(feature_2)
        feature_3_minMax = DP.minMaxData(feature_3)
        feature_mix_minMax = DP.concat_multiple_dfs(feature_1_minMax, feature_2_minMax, feature_3_minMax)
        feature_mix_minMax = DP.pcaData(feature_mix_minMax, dimension)
        return MODEL.gmm_clustering(feature_mix_minMax, attemptTime, 4)
    # 计算GMM_normalize
    if initMethod == 'normalize':
        feature_1_normalize = DP.minMaxData(feature_1)
        feature_2_normalize = DP.minMaxData(feature_2)
        feature_3_normalize = DP.minMaxData(feature_3)
        feature_mix_normalize = DP.concat_multiple_dfs(feature_1_normalize, feature_2_normalize, feature_3_normalize)
        feature_mix_normalize = DP.pcaData(feature_mix_normalize, dimension)
        return MODEL.gmm_clustering(feature_mix_normalize, attemptTime, 4)
    return None

def ratio_hierarchical(initMethod, feature_1, feature_2, feature_3,dimension):
    # 计算Hierarchical_standardize
    if initMethod == 'standardize':
        feature_1_standardize = DP.standardizeData(feature_1)
        feature_2_standardize = DP.standardizeData(feature_2)
        feature_3_standardize = DP.standardizeData(feature_3)
        feature_mix_standardize = DP.concat_multiple_dfs(feature_1_standardize, feature_2_standardize,feature_3_standardize)
        feature_mix_standardize = DP.pcaData(feature_mix_standardize, dimension)
        return MODEL.hierarchical_clustering(feature_mix_standardize, n_clusters=4)
    # 计算Hierarchical_minMax
    if initMethod == 'minMax':
        feature_1_minMax = DP.minMaxData(feature_1)
        feature_2_minMax = DP.minMaxData(feature_2)
        feature_3_minMax = DP.minMaxData(feature_3)
        feature_mix_minMax = DP.concat_multiple_dfs(feature_1_minMax, feature_2_minMax, feature_3_minMax)

        feature_mix_minMax = DP.pcaData(feature_mix_minMax, dimension)
        return MODEL.hierarchical_clustering(feature_mix_minMax, n_clusters=4)
    # 计算Hierarchical_normalize
    if(initMethod == 'normalize'):
        feature_1_normalize = DP.minMaxData(feature_1)
        feature_2_normalize = DP.minMaxData(feature_2)
        feature_3_normalize = DP.minMaxData(feature_3)

        feature_mix_normalize = DP.concat_multiple_dfs(feature_1_normalize, feature_2_normalize, feature_3_normalize)

        feature_mix_normalize = DP.pcaData(feature_mix_normalize, dimension)
        return MODEL.hierarchical_clustering(feature_mix_normalize, n_clusters=4)
    return None