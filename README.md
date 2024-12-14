# mlproject2

Results with scaling
5-fold Cross Validation with Random Seed 6

                      Model  Mean Accuracy  Std Deviation
0       Logistic Regression       0.644094       0.119027
1  Decision Tree Classifier       0.762055       0.178205
2  Random Forest Classifier       0.731037       0.133652
3                       SVM       0.696673       0.139834
4                   XGBoost       0.762055       0.178205

Results without scaling
5-fold Cross Validation with Random Seed 6
                      Model  Mean Accuracy  Std Deviation
0       Logistic Regression       0.711073       0.084569
1  Decision Tree Classifier       0.942094       0.034922
2  Random Forest Classifier       0.884745       0.076068
3                       SVM       0.579220       0.075473
4                   XGBoost       0.942094       0.034922


Suggestions:
- Tree-Based Models (e.g., Decision Tree, Random Forest, XGBoost): These models are not sensitive to the scale of the input features because they split data based on feature thresholds, not on distances or magnitudes. Applying standard scaling can distort the original structure of the data, which might reduce their performance.
- This explains why the Decision Tree Classifier, Random Forest Classifier, and XGBoost performed worse after scaling.
- Do Not Scale Tree-Based Models: Avoid applying standard scaling for Decision Tree, Random Forest, and XGBoost models, as it doesn't benefit them.
- Investigate Outliers: Outliers can disproportionately affect standard scaling. Use robust methods to handle outliers or scale robustly if needed.

5-fold Cross Validation with Random Seed 42
                      Model  Mean Accuracy  Std Deviation
0       Logistic Regression       0.604162       0.087575
1  Decision Tree Classifier       0.942094       0.034922
2  Random Forest Classifier       0.897746       0.066374
3                       SVM       0.696673       0.139834
4                   XGBoost       0.942094       0.034922

5-fold Cross Validation with Random Seed 26

                      Model  Mean Accuracy  Std Deviation
0       Logistic Regression       0.604162       0.087575
1  Decision Tree Classifier       0.942094       0.034922
2  Random Forest Classifier       0.908244       0.057533
3                       SVM       0.696673       0.139834
4                   XGBoost       0.942094       0.034922