Human Activity Recognition using Random Forest Prediction Model
========================================================
### Data
This exercise is aimed at predicting the class of exercise performed by users based on data captured using wearable accelerometers.
AHR dataset for this study is provided by Groupware (http://groupware.les.inf.puc-rio.br/har)

### Exploratory Analysis
The raw dataset not only contained the accelerometers measurements but also computed statistics (Variance, Standard Devation, skewness, Kurtosis, Max, Min, Average) for each of the observation.

For the purpose of this analysis, we remove all columns containing computed statistics from the dataset to ensure we have correct set of mathematically independant predictors.

Outcome variable, classe, is a factor with 5 outcomes therefore, for creating the model, we need to use Random Forest technique (since GLM can be used only if outcome variable is a factor with only 2 outcomes).

### Fitting the Model
We randomly sample 3000 records from the entire training set of 19,622 records (on account of limited computational resources available for the purpose of this exercise) and fit a training model using Random Forest method on these 3000 records.

The accuracy predicted by the model is 100% while that from Confusion Matrix is 96.7%

### Cross Validation
For cross validation, the training model fitted above is used to predict the outcome (classe) for the entire dataset of 19,622 records. The predicted value of classe is compared with the actual value availale in the raw dataset. The observed prediction accuracy rate for the model fitted by us is 98.05%

### Predicting for the Test Dataset
Finally, we use our fitted model on the testing dataset of 20 records, and append predicted values to the dataset.


### Load required libraries

```r
suppressWarnings(library(caret))
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```


### Download Raw Training and Testing Data

```r
setInternet2(TRUE)
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    destfile = "pml-training.csv")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
    destfile = "pml-testing.csv")
```


### Load data in csv files into datasets

```r
pmltraining <- read.csv("pml-training.csv", header = TRUE)
pmltesting <- read.csv("pml-testing.csv", header = TRUE)
```




### Remove columns containing mathematically/statistically calculated values and NAs from the training dataset

```r
pmltraining <- pmltraining[, -grep("^var", colnames(pmltraining))]
pmltraining <- pmltraining[, -grep("^avg", colnames(pmltraining))]
pmltraining <- pmltraining[, -grep("^stddev", colnames(pmltraining))]
pmltraining <- pmltraining[, -grep("^skew", colnames(pmltraining))]
pmltraining <- pmltraining[, -grep("^kurtosis", colnames(pmltraining))]
pmltraining <- pmltraining[, -grep("^max", colnames(pmltraining))]
pmltraining <- pmltraining[, -grep("^min", colnames(pmltraining))]
# Remove columns with NAs
pmltraining <- pmltraining[sapply(pmltraining, function(x) !any(is.na(x)))]
# Remove columns with blank values
pmltraining <- pmltraining[, -grep("^amplitude", colnames(pmltraining))]
# Remove first 5 columns since they don't contribute to effect
pmltraining <- pmltraining[, -c(1:5)]
```



### Remove columns containing mathematically/statistically calculated values and NAs from the testing dataset

```r
pmltesting <- pmltesting[, -grep("^var", colnames(pmltesting))]
pmltesting <- pmltesting[, -grep("^avg", colnames(pmltesting))]
pmltesting <- pmltesting[, -grep("^stddev", colnames(pmltesting))]
pmltesting <- pmltesting[, -grep("^skew", colnames(pmltesting))]
pmltesting <- pmltesting[, -grep("^kurtosis", colnames(pmltesting))]
pmltesting <- pmltesting[, -grep("^max", colnames(pmltesting))]
pmltesting <- pmltesting[, -grep("^min", colnames(pmltesting))]
# Remove columns with NAs
pmltesting <- pmltesting[sapply(pmltesting, function(x) !any(is.na(x)))]
# Remove first 5 columns since they don't contribute to effect
pmltesting <- pmltesting[, -c(1:5)]
```



### Randomly sample 3000 records from raw training dataset to match the available computing power/resources

```r
set.seed(1234)
trainInds <- sample(nrow(pmltraining), 3000)
train <- data.frame(pmltraining[trainInds, ])
```


### Fit the training model and validate its accuracy level

```r
fitMod <- suppressWarnings(train(classe ~ ., data = train, method = "rf"))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
fitMod
```

```
## Random Forest 
## 
## 3000 samples
##   54 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 3000, 3000, 3000, 3000, 3000, 3000, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         0.9    0.008        0.01    
##   30    1         1      0.007        0.009   
##   50    1         0.9    0.01         0.01    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```

```r
confusionMatrix(fitMod)
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.1  0.6  0.0  0.0  0.0
##          B  0.0 18.1  0.7  0.1  0.2
##          C  0.0  0.4 17.8  0.6  0.1
##          D  0.1  0.0  0.2 15.3  0.2
##          E  0.0  0.0  0.0  0.1 17.4
```


### Perform cross validation of model's accuracy with the original raw data

```r
predfit <- predict(fitMod, pmltraining)
pmltraining$predRight <- predfit == pmltraining$classe
table(predfit, pmltraining$classe)
```

```
##        
## predfit    A    B    C    D    E
##       A 5570  107    0    5    0
##       B    8 3626   86    4    3
##       C    0   61 3327   60    1
##       D    1    3    9 3129   21
##       E    1    0    0   18 3582
```


### Predict outcome for the 20 test cases using the model fitted above and append the predicted outcome to the testing data. Save the results in a new 'output' dataset. Create a .csv file with predictions for the testing dataset

```r
predictions <- predict(fitMod, pmltesting)
pmltesting$predRight <- predictions
output <- cbind(pmltesting, predictions)
write.table(output, "Predictions.csv")
```

