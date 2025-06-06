import DataProcessing as DP
import pandas as pd

# 首先，划分feature和labels
data = pd.read_excel('training_set.xlsx')
features, labels = DP.getFeaturesAndLabels(data, 'Gender')

# 数据预处理和划分
# 80%特征（用于训练），80% 正确标签，20%特征（用于验证），20%正确标签
features_train, labels_train, features_test, labels_test, scaler = DP.preprocessAndSplit(features, labels, 0.2,1)

print("==================================================================================================")
# KNN 实验
knn_best_model = 0
knn_best_accuracy = 0
for n_neighbors in range(1,100):
    knn_current_info = DP.train_knn_classifier(features_train, labels_train,features_test, labels_test, n_neighbors=n_neighbors)
    if knn_current_info[1] > knn_best_accuracy:
        knn_best_model = knn_current_info[0]
        knn_best_accuracy = knn_current_info[1]
print(f"\033[1;31;40m{f"[KNN BEST MODEL IS n_neighbors = {knn_best_model.n_neighbors}  WITH ACCURACY = {knn_best_accuracy:.2f}]"}\033[0m")
print("==================================================================================================")
# 决策树实验
dt_model_gini     = DP.train_decision_tree_classifier(features_train, labels_train,features_test, labels_test, dt_criterion='gini')
dt_model_logLoss  = DP.train_decision_tree_classifier(features_train, labels_train,features_test, labels_test, dt_criterion='log_loss')
dt_model_entropy  = DP.train_decision_tree_classifier(features_train, labels_train,features_test, labels_test, dt_criterion='entropy')
print("==================================================================================================")
# 朴素贝叶斯实验
nb_model = DP.train_naive_bayes_classifier(features_train, labels_train,features_test, labels_test)
print("==================================================================================================")
# 集成学习实验
ensemble_model = DP.train_ensemble_classifier(features_train, labels_train,features_test, labels_test,
                                              knn_best_model, dt_model_logLoss, nb_model, ec_voting='hard')
ensemble_model_soft = DP.train_ensemble_classifier(features_train, labels_train,features_test, labels_test,
                                                   knn_best_model, dt_model_logLoss, nb_model,
                                                   ec_voting='soft')
print("==================================================================================================")