## Supervised Learning Methods

### Desicion Tree

### Random Forest

* Run several models at the same time, and compete against with each other

* ### 优点：

  1. **高准确性**：通过集成多个决策树的结果，随机森林通常比单个决策树有更高的预测准确性。
  2. **防止过拟合**：随机森林通过Bagging和随机特征选择，减少了单个决策树过拟合的风险。
  3. **处理高维数据**：随机森林可以处理大量的输入变量，且不需要特征选择。
  4. **抗噪声能力强**：在数据集包含较多噪声时，随机森林仍能保持良好的性能。
  5. **处理缺失值**：随机森林能够处理数据中的缺失值。
  6. **输出特征重要性**：可以评估各个特征的重要性，提供对模型的解释能力。

* The random are about:
  * Get the tree
    * random select rows
    * select columns to select the best slipt
    * repeat step 2 till tree is get
  * Get several trees

### Support Vector Machines



我想做generlization error 越低越好 ==》 overfitting

所有数据跑完以后，和

confusion matrix (画一个)



我对数据做了什么来防止over fit （大章） k-fold

feature select -rfe （筛选掉ml认为不相关的）

results test sets

![image-20240622154557557](C:\Users\29505\AppData\Roaming\Typora\typora-user-images\image-20240622154557557.png)





### Random forest:

TRAIN
Accuracy: 0.8640858999717435
Precision: 0.8650323377116125
Recall: 0.8640858999717435
VALIDATION
Accuracy: 0.8122171945701357
Precision: 0.8098244306389102
Recall: 0.8122171945701357
TEST
Accuracy: 0.7584650112866818
Precision: 0.7505067912737355
Recall: 0.7584650112866818

### Random forest with optimisted data:

Best parameters found:  {'max_depth': 20, 'min_impurity_decrease': 0.0, 'min_samples_split': 0.01, 'n_estimators': 80}
Best estimators found:  RandomForestClassifier(max_depth=20, min_samples_split=0.01, n_estimators=80,
                       random_state=0)

TRAIN

Accuracy: 0.8671941226335123
Precision: 0.8688400939965787
Recall: 0.8671941226335123
VALIDATION
Accuracy: 0.8076923076923077
Precision: 0.8070247023444875
Recall: 0.8076923076923077
TEST
Accuracy: 0.7720090293453724
Precision: 0.767241807764636
Recall: 0.7720090293453724

### Random forest with optimisted data and rfe:

rfe = 35 -> 20, step = 3

TRAIN
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
VALIDATION
Accuracy: 0.832579185520362
Precision: 0.8296124127345534
Recall: 0.832579185520362
TEST
Accuracy: 0.7945823927765236
Precision: 0.7886728165640592
Recall: 0.7945823927765236
Total runtime of the script: 27.8073947429657 seconds

rfe = 35 -> 25

TRAIN
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
VALIDATION
Accuracy: 0.8371040723981901
Precision: 0.8345101354151128
Recall: 0.8371040723981901
TEST
Accuracy: 0.7990970654627539
Precision: 0.7932993741448892
Recall: 0.7990970654627539



Xgboost:

Train time:

1971.6269881725311 seconds





做一个纯随机猜

UMAP (数据不够好)，发现数据不是完全区分