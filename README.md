**Diagnosis of heart disease from the Cleveland Heart Disease dataset**

**1.INTRODUCTION**

According to the World Health Organization, about 17 million deaths occur worldwide each year due to heart disease, accounting for 30% of the deaths that occur. The mortality rates of cardiovascular diseases have been increasing rapidly all over the world in recent years. Four out of every five heart disease deaths are due to strokes and heart attacks. Therefore, heart disease should be predicted in its early stages. Identifying the most influential factors of heart disease and accurately estimating and early diagnosing overall risk plays a vital role in making decisions about lifestyle changes in high-risk patients and thus reduces complications. Diagnosis and treatment of heart disease is extremely complex, especially in developing countries where diagnostic tools, other resources that affect the correct prognosis of doctors, are limited. People at risk of heart disease may also have symptoms such as high blood pressure, obesity, cholesterol, diabetes, age, etc.

In recent years, the health system has collected a huge amount of data on heart disease, and datasets have been created that consist of different medical parameters or characteristics, such as age, gender, blood pressure, cholesterol, etc. It has been seen in the studies that the use of machine learning algorithms together with the correct analysis of these data sets gives very successful results.

**2.PROBLEM DESCRIPTION**

The biggest challenge in heart disease is the detection of heart disease. The fact that there are many different parameters such as diabetes, cholesterol, blood pressure, which trigger cardiovascular diseases, creates difficulties in the diagnosis of the disease. There are existing tools that can predict heart disease, but they are either expensive or not sufficient in calculating the likelihood of heart disease in humans. Early detection of heart disease reduces mortality and overall complications. However, it is not possible for patients to be followed by a doctor 24 hours a day, as it requires more time and expertise. Today, since there is a sufficient amount of data and suitable for its purpose, machine learning algorithms are very popular and useful in the field of health, especially in the diagnosis of diseases.

The aim of the project is to determine whether patients have heart disease and if so, its degree (1-4) by using different machine learning algorithms (such as SVM, KNN, ANN) together with some characteristics given to the patients. It is to find the algorithm that gives the highest accuracy.

**3.DATASET and METHODS**

The Heart Disease dataset from Cleveland, Hungary and VA Long Beach in Kaggle was used to classify whether patients had heart disease or not. The dataset contains 920 observations, 15 features (id, age, origin, sex, chest-pain type(cp), resting blood pressure(trestbps), serum cholesterol(chol), fasting blood sugar(fbs), resting ECG(restecg), thalach, exercise induced angina(exang), oldpeak, slope, ca, thal) and 1 target(num).

The Target field indicates the presence of heart disease in the patient and is an integer between 0-4. 0 indicates that there is no cardiovascular disease, that is, less than 50% of vascular narrowing, 1-4 cardiovascular diseases, that is, more than 50% of vascular narrowing and their degree.

The table shows the types, descriptions, and categories of features included in the dataset:

| Attribute | Type    | Categories                                                             |
|-----------|---------|------------------------------------------------------------------------|
| İd        | İnteger | 1-920                                                                  |
| Age       | İnteger | 28-77                                                                  |
| Sex       | Object  | Female(0)/Male(1)                                                      |
| Dataset   | Object  | Cleveland(0)/Hungary(1)/Switzerland(2)/VA Long Beach(3)                |
| Cp        | Object  | typical angina(3), atypical angina(1), non-anginal(2), asymptomatic(0) |
| Trestbps  | Float   | 0-200                                                                  |
| Chol      | Float   | 0-603                                                                  |
| Fbs       | Object  | True(1)/False(0)                                                       |
| Restecg   | Object  | Normal(1), stt abnormality(2), lv hypertrophy(0)                       |
| Thalch    | Float   | 60-202                                                                 |
| Exang     | Object  | True(1)/False(0)                                                       |
| Oldpeak   | Float   | -2,6-6,2                                                               |
| Slope     | Object  | Upsloping(2), flat(1), downsloping(0)                                  |
| Ca        | Float   | 0.0, 1.0, 2.0, 3.0                                                     |
| Thal      | Object  | Normal(1), fixed defect(0), reversible defect(2)                       |
| num       | integer | 0,1,2,3,4                                                              |

![](media/1f06749f7260afb88ab1a801c44ba421.png)

Distribution of dataset target values:

![](media/a62aea2564fa7320eacc1d138a163b90.png)![](media/144db4605cac31f91aceb5866322a2c3.png)

**3.1. Data Preprocessing**

Dataset reviews were done using Python's Pandas library. The dataset has some categorical variables, null and outliers, that need to be converted to numeric values.

First of all, Null values were determined. The dataset has Null values in the trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal columns.

**3.1.1.Detection and Filling of Null Values**

The Null values in the oldpeak, chol, trestbps and thalch columns are filled with the median of these columns, while the remaining categorical columns are filled with the most repeating values in these columns. Null values are eliminated.

![](media/acafe62dc225e644002026db71ecf798.png) ![](media/1fee6b324d2fe9d40e57e23de82b896e.png)

In order to better analyze the numerical variables in the data set, the following output was obtained by using the describe method.

![](media/26456172475190e796fe54972a8ff4cc.png)

**3.1.2. Detection and Removal of Contradictory Value**

There are outliers in the Trestbps, chol, thalch and oldpeak columns. Outliers have been replaced by the maximum and minimum threshold values in the columns they have.

![](media/ae036f0e25458b19f83d8ebbc7450972.png)

Below are graphs with values for columns with outliers on the left, and charts with cleared outliers are shown on the right.

![](media/5896d74fae9dc337c68b364f2c248707.png)![](media/63bf7997c71458023a17ae01d8515338.png)

![](media/fb19572e3c733bec82261835edd3faf6.png)![](media/ad91f7b80db8db7799c7327bf9ed6ea6.png)

![](media/a681f230d55cddc28a0d21086a9a9dcf.png)![](media/c1a83c9b3b8f8554f5121cd33f8e4523.png)

![](media/5e2efd5b5264c00e07c34e8d24786ac3.png)![](media/6cbbedc294fa51e6b811e84e4f14c784.png)

**3.1.3. Label Encoding and Correlation Matrix**

Label encoding serves to digitize data. That is, it assigns a numeric value to each categorical data. With this process, categorical values are converted to numeric values.

![](media/06d89ccc257edbec096f6b5e5f3e201e.png)

Correlation Matrix is a statistical method that shows the relationship between two or more variables. It helps to define the relationship and dependence between variables. The correlation matrix of the dataset is as follows:

The correlation matrix showed that there is no feature with a very high correlation with the target value. In addition, some of the attributes have a negative correlation with the target value, and some have a positive correlation.

![](media/a156a37a5f976692a91bbefa00693267.png)

Feature Importance graph was obtained with SelectKBest algorithm. Exang, oldpeak, thalch, cp, age, dataset columns are the first 5 columns that affect the target variable.

![](media/4a6a2e139e5228c63e09230fda46127b.png)

**3.2. Models**

**3.2.1. Models Created by Hold Out Method**

For the preparation stages of the model, the dependent variables were assigned to x and the independent variable to y. x and y sets are divided into 30% test and 70% train. X_train and X_test sets have been standardized with the StandardScaler algorithm.

![](media/d1204a4ed911f0aa35295255c4094bde.png)

**3.2.1.1. K Nearest Neighbor (KNN)**

![](media/ba12bad11887b24d91990ced90737039.png)

It is a supervised machine learning method in which the class where the sample data point to be classified is located and the nearest neighbor (element) are determined according to the k value. The KNN algorithm is a classification algorithm that makes class predictions based on the knowledge in which class the nearest neighbors of the matrix are denser, in which the independent variables of the value to be predicted are located. It makes a prediction based on two basic values, distance and number of neighborhoods (K). Distance is the distance of the point to be predicted from other points. For this, the Euclidean Distance was used. The K value is chosen as 15.

The purpose of using the KNN algorithm in this study is to predict and classify which class the new data belongs to. The algorithm creates a close-range ranking by comparing the newly arrived data with the distance of Euclidean distance with the distances of all other data, takes the nearest neighbors in the order as much as the given K value and works by estimating which class it belongs to, the more dense the new incoming data is.

**3.2.1.2. Support Vector Machine (SVM)**

![](media/fa6f93b057d083acf7c1303167c0b82a.png)

Support Vector Machine (SVM) are powerful and flexible supervised machine learning algorithms used for both classification and regression. They are generally used in classification problems. The primary role of the SVM algorithm is to separate the two classes by forming a hyperplane line. The data points closest to the hyperplane are known as support vectors. The goal is to select the hyperplane with the maximum possible margin between them and any point in the training set, which increases the likelihood of a new data being properly classified. The main purpose of SVM is to find a hyperplane in N-dimensional space that will classify all data points. In the SVM method, "rbf" is used as the distance kernel. Since it is a multi-class classification, decision_function_shape = "ovo" is chosen.

**3.2.1.3. Artificial Neural Network (ANN)**

![](media/b8898a1cd2ee0e8a51548cfd2510a5d3.png)

Another method used in classification is Artificial Neural Networks (ANN). Artificial neural networks are an information-processing technology inspired by the human brain's skills such as learning and creating new information. It is the result of mathematical modeling of learning by taking the human brain as an example. The learning process in artificial neural networks is carried out through examples. During learning, input and output information is given and rules are made. Artificial neural networks are made up of neurons. The brain is an extremely complex, non-linear, and parallel computer system. Structural components known as neurons can organize certain calculations faster than the fastest digital computer in existence. An artificial nerve cell consists of five parts: inputs, weights, addition function, activation function, and outputs. The inputs are the data coming to the neuron multiplied by the weight of the connections from which they come and sent to the nucleus. The Total function calculates the net input of the cell by adding the inputs by multiplying them by weights. The action function takes the weighted sum of all inputs in the previous layer, produces an output value, and passes it to the next layer.

The MLPClassifier algorithm was used for the ANN algorithm. The activation function is "relu", which is the default. (rectified linear unit function, f(x) = max(0, x)). "lbfgs was chosen as Solver because it can converge faster to small data sets and perform better.

**3.2.2. Models Created by Stratified K-Fold Method**

Because the target variable classes in the dataset do not have instances of equal numbers, the x and y data sets are separated by the Stratified K-Fold method. N_split=5 is selected.

**3.2.2.1.K Nearest Neighbor (KNN)**

When the most optimal k value was looked at from the graph, k = 15 was found. "Euclidean distance" is used for distance calculation. The K value is chosen as 15.

![](media/a7ff61c1c7a84d68f4ca2559aeed0b32.png)

![](media/d41101742660871a6ff39d4a225692a0.png)

**3.2.2.2. Support Vector Machine (SVM)**

In the SVM method, "rbf" is used as the distance kernel. Since it is a multi-class classification, decision_function_shape="ovo"ischosen.![](media/db6af380239deb6fa30b41b5c9915eb3.png)

**4.RESULTS**

**4.1. Hold Out Method Results**

**4.1.1. KNN**

**4.1.1.1 Multi Class Classification**

In the KNN method, "Euclid" is chosen as the distance measurement method and 15 is chosen as the k value. The accuracy achieved for 5 different classes was 59%.

**![](media/6d15bff789a6f50860683cacab5b2ba4.png)![](media/927c7be5a2d717ca015e299639372f93.png)**

**![](media/8bbf7d319f4bf569902fa8f868a20a59.png)**

**4.1.2.2 Binary Classification**

In the KNN method, "Euclid" is chosen as the distance measurement method and 15 is chosen as the k value. The accuracy achieved for 2 different classes was 83%. The target values of 2, 3, and 4 have been changed to 1.

**![](media/1ed5978c086a5fe66e10a559af1d5657.png)![](media/15aef64e39658d4028c0af26a9f9bcf6.png)**

**![](media/ac97c1d78f79903666bb66d53b338307.png)**

**4.1.2. SVM**

**4.1.2.1 Multi Class Classification**

In the SVM method, "rbf" was used as the distance kernel. the accuracy obtained for 5 different classes was 56%. Since it is a multi-class classification, decision_function_shape = "ovo" is chosen.

**![](media/0bf0c13a49bb22a89fb9cf8aa925a9a3.png)![](media/0a2b9e48297977b92390f7c1516eac79.png)**

**![](media/1681ec4a68a319199f052be7a3701a1b.png)**

**4.1.2.2 Binary Classification**

In the SVM method, "rbf" is used as the distance kernel. The accuracy achieved for 2 different classes was 85%. The target values of 2, 3, and 4 have been changed to 1.

**![](media/5fda8f036ee711b13a7e060d8aa10882.png)![](media/0000947ad0eb523034f04a511059032e.png)**

**![](media/aba06f49c9288815826f9212618bd355.png)**

**4.1.3 ANN**

**4.1.3.1 Multi Class Classification**

MLPClassifier algorithm was used in ANN method. A 4-layer hidden layer has been created. The accuracy achieved for 5 different classes was 57%.

**![](media/079cb83a862709619c0562ad7516ec40.png)** **![](media/aa79ed9bdfb921bb0d8f2921a5d9ece9.png)**

**![](media/608c8cd531490ba4383649ee03d71b9c.png)**

**4.1.2.2 Binary Classification**

MLPClassifier algorithm was used in ANN method. A 4-layer hidden layer has been created. The accuracy achieved for 2 different classes was 72%. The target values of 2, 3, and 4 have been changed to 1.

**![](media/cd9e946ffcabe5dee99346daec438322.png)![](media/b8ec5cfc6a2e9ea8d405a669c47f3203.png)**

**![](media/e97d7ff137b84803db2b41b7bb74b39c.png)**

**4.2. Results of the Stratified K-Fold Method**

**4.2.1. KNN**

**4.2.1.1 Multi Class Classification**

In the KNN method, "Euclid" is chosen as the distance measurement method and 15 is chosen as the k value. The accuracy achieved for 5 different classes was 60%.

**![](media/0f708f626d0d43bb3c4f15b012be7b9d.png)![](media/9b1ab4d080ee6249a54bffb87b30616d.png)**

**4.2.1.2 Binary Classification**

In the KNN method, "Euclid" is chosen as the distance measurement method and 15 is chosen as the k value. The accuracy achieved for 2 different classes was 85%. The target values of 2, 3, and 4 have been changed to 1.

**![](media/67c2c6a1669f37fdf02971b211b39e9c.png)![](media/733deb6d678c7b58dfd69f05be63d64e.png)**

**4.2.2. SVM**

**4.2.2.1 Multi Class Classification**

In the SVM method, "rbf" was used as the distance kernel. the accuracy obtained for 5 different classes was 64%. Since it is a multi-class classification, decision_function_shape = "ovo" is chosen.

**![](media/f2890bba01aa04ccfe4520c43fe9f331.png)![](media/03901faad4bfbc1794c3d36b9886c32a.png)**

**4.2.2.2 Binary Classification**

In the SVM method, "rbf" is used as the distance kernel. The accuracy achieved for 2 different classes was 89%. The target values of 2, 3, and 4 have been changed to 1.

**![](media/042c623f4c18fa7752848c81b6aa4fe9.png)![](media/9448f9d1fbd83ea80bda74dfe1df9ed2.png)**

**5.DISCUSSION**

The mortality rates of heart diseases are quite high, and early and accurate diagnosis is very important for patients. This project is related to the determination of heart disease from the values such as age, gender, blood pressure of the patients. In this direction, first of all, missing values and outliers were detected and filled in the data set. A correlation matrix was drawn and the rates at which the attributes affected the target variable were examined. After the label encoding process, the data set became suitable for use in machine learning algorithms.

Data set test train separation is made in 2 different ways. In the 1st separation, the data set is divided into 30% test and 70% train by hold out method. In the 2nd separation, the Stratified K Fold method was preferred due to the imbalance in the data set, and 5 was selected as the n_split to be suitable for the data set size.

Classification studies were carried out with KNN, SVM and ANN as machine learning algorithms. For the hold out method, the maximum accuracy was achieved in SVM 85% in binary classification and KNN 59% in multi class classification. For the Stratified K Fold method, maximum accuracy was achieved in 89% SVM in binary classification and 64% in multi class classification.

The dataset contains 411 examples of target values, 265 from 1, 109 from 2, 107 from 3, and 28 from 4. The data set distribution is uneven. For this reason, the accuracy obtained in multi-class classifications is quite low compared to binary classification.

Since the data set is not evenly distributed, the Stratified K-fold method was preferred to separate the data set. In this method, instead of random sampling, the accuracy was higher than the hold out method because the separation process was performed according to the weight of the sample amount in the data set.

Because the ANN algorithm is more sensitive to noise and unbalanced data, accuracy is thought to be adversely affected. SVM and KNN accuracies were close to each other. This is thought to be due to the fact that both algorithms are more tolerant of noise and imbalance in the data set. The available data in this dataset are considered insufficient for 5 different outcome classifications.

Compared to other studies, it is seen that my work has higher accuracy values.

| Name of the Work                            | Classification Type  | Algorithms Used  | Results       |
|---------------------------------------------|----------------------|------------------|---------------|
| My Study (Hold Out)                         | (920,16). Multiclass | KNN, SVM, ANN    | %59, %56, %57 |
| My Study (Hold Out)                         | (920,16). Binary     | KNN, SVM, ANN    | %83, %85, %72 |
| My Study (Straitified K Fold)               | (920,16). Multiclass | KNN, SVM         | %60, %64      |
| My Study (Straitified K Fold)               | (920,16). Binary     | KNN, SVM         | %85, %89      |
| Heart Rate Prediction By Using 5 Models [2] | (920,16). Binary     | KNN, SVM, ANN    | %53, %56, %85 |
| Heart_Disease_Prediction_Task1 [3]          | (920,16). Multiclass | KNN, SVM         | %50, %53      |
| UCI_Heart Disease [4]                       | (920,16). Multiclass | KNN, ANN         | %39, %40      |

As future studies, accuracy can be looked at with different machine learning algorithms and variables that are high in feature importance.
