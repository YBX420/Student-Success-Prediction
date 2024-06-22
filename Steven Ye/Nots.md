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



我对数据做了什么来防止over fit （大章）

feature select -rfe （筛选掉ml认为不相关的）

results test sets







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

做一个纯随机猜

UMAP (数据不够好)，发现数据不是完全区分