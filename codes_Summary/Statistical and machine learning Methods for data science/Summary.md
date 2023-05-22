# CH:1 INTRODUCTION TO DATA SCIENCE

Data science comprises of mathematics and statistics, computer science and domain knowledge
other important areas are: Communication, visualization, hard and soft skills.

## Mathematics and Statistics
Data scientists need to have mathematics and statistics skills to understand the data available, prepare the data needed to train a model, deploy multiple approaches in training and validating thr analytical model, assess the model's results and explain and intepret the model's outcome.

- it is needed to summarize data to describe past events (Descriptive statistics)
- It takes the results of a sample and generalize to a larger population (Inferetial Statistics)
- The skills are needed to fit models where the responsive variable is known, and based on that , train a model to classify, predict or estimate future outcomes. (Supervised modeling)
- It is used in optimization

Math and statistics are needed when business conditions require a specific event and there is no past behaviour to drive the training of a supervised model. The learning is based on discovering previously unknown patterns in the dataset (Unsupervised modeling)

## Computer Science
Data scientists do not develop models by hand due to the amount of volume of data. 
They need some level of computer skills to develop, code, exxtract, prepare, transform, merge, and store data, assess model results and deploy models in production. All these steps are performed in a digital environment using distinct frameworks, languages and storage. 
(cloud-based computing)

## Domain Knowledge
It is important to understand the problem and evaluate what is necessary to solve it
It is important to understand how to link the model results to a practical action
The analytical model gives direction. How to use the results to really solve the problem, based on a series of information, policies, regulations,impacts and so on is the key factor to success.
Awareness of the business problem and how it affects companies, communitoes, governments and people.

Collaboration is another key factor in data science.

## Communnication and Visualization
The results need to be communicated. This commmunication can involve to explain and intepret models.
Data scientists communicate how the results can be used to improve the operational processwith the business side of the company. They use visula presentation expertise and story-telling capabilities to cretae an appealing story on how the model's results can be applied to business problems.

## Hard and Soft skills
Hard skills include: Math, statistics, computer science, data analysis , programming etc
Soft skills include: Problem-solving skills, communication, curiosity, innovation, storytelling etc...

## DATA SCIENCE APPLICATION  
Data analysis and analytical insights replace guessing and assumptions about market, customers and business scenarios.
Industries that can benefit from data science include: Retail, communication, banking, energy, insurance, government etc..
Data science helps in solving the following business issues: Fraud detection, churn analysis, bad debt, loss estimation, cross-sell/up-sell, risk analysis, segmentation, collecting, optimization, forecasting, supply chain, and anomaly detection among others

## DATA SCIENCE LIFECYCLE AND THE MATURITY FRAMEWORK

### Underatnad the question
Understand what you are trying to solve with the model you are about to develop.
check if the model is appropriate, if it is feasible, if there is enough data to use and what practical actions are planned to be deployed based on the model's outcome.

QUESTIONS TO ASK:
- what is the goal?
- what is the specific objective for the model?
- is there enough data to address the problem?
- what actions are planned based on the model's outcom?

### Collect Data
This requires multiple people and different skills. Database management, repositories, programming, data quality packages, data integration and many other technilogies.

QUESTIONS TO ASK:
- What data is relevant?
- How many data sources are involved? 
- Where do the data sources reside?
- Is the access to the data readily available?
- are there any privacy issues?
- are the data available when the model is deployed in production?

### Explore the Data
Explore and evaluate the quality and appropriateness of the information available.
This involves: Data analysis, cardinality analysis, data distribution, multivariate analysis and data quality analyses.
verify if the data is available in the correct format.

QUESTIONS TO ASK:
- what anomalies or patterns are noticeable in the data set
- are there too many variable to create the model
- are there too few variables to ceate the model
- are tasks assigned to  create new inputs
- are data transformations required to adjust the input data from the model training
- are tasks assigned to reduce the number of inputs

### Model the Data
Data scientists will use their creativity and innovation skills to try out multiple analytical approaches to solve the business problem

QUESTIONS TO CONSIDER:
- which model has the highest predictive accuracy?
- which model best generalizes the new data?
- is it possible to test/validate the model?
- which model is most is the modt intepretable?
- which model best explains the correlation between the input variables and the target?
- which one best describes the effects of the predictors to the estimation?
- which model best addresses the business goals?

Data scientists use different algorithms,  techniques and distinct analytical approaches.
Data scientists try to fit the model on a portion of the data and evaluate the model's performance on another part of the data. The first portion is the **training** set and the secon one is the **validation** set. and occasionally a third portion called **test** set.

Modeling Data includes the following tasks:
- training different models(algorithms/techniques/analytical approaches)
- validate all trained models based on a different data set.
- Test all models trained on different data sets (different from one used in validation)
- assess all models' outcomes and evaluate the results based on the business goals.
- select the best model based on the business requirements.
- Deploy and score the best model to support the business action required.

### Provide an Answer
provide answers to the questions raised and validated.

QUESTIONS To ask:
- what lessons were learned from the reaining models?
- how do the trained models answer the original questions?
- How do the reained models tell a story and support business decisions?
- How can the trained models be used and deployed in production in the appropriate timeframe to support the required business action?

Once the answer is provided it might generate more questions regarding the business problem. Therefore, data science is cyclical as the process is repeated until the problem is solved 

## ADVANCED ANALYTICS IN DATA SCIENCE
This field encompasses more than just math models and statistical analysis, it comprises of machine learning, forecasting, text analytics and optimization.

There are 2 types of machine learning models: 
- supervised: when the response variable (target) is known and used in the model. examples of supervised learning are: Regression, decision tree, random forest, gradient boosting, neural networks and support vector machine.
- unsupervised: when the target is not known. examples of unsupervised models are: clustering, association rules, sequence association rules, path analysis, link analysis.

There is a variation of these types of models called **semi-supervised** which involves a small amount of data where the target is known and a large amount of data where the target is unkown

There are models associated with **reinforcement learning** where the algorithm is trained by using a system to reward the step when the model goes in the right direction and punish the step that when the model goes in the wrong direction.

Machine Learning models automatically improve the learning steps based on the input and the objective function.

Data scientists should be able to build statistical and machine learning flow to access the available data, prepare the data for supervised and unsupervised modeling, fit different types if analytical models to solve the business problems, evaluate models according to the business requirements and deploy the champion model and challenger models in production to support the planned business actions.

Machine learning models  can be more accurate and perform better in production to support business actions but can be hard to intepret and explain.
Statistical models can generalize better estimations for future events, they are oten simpler and easier to intepret and explain.

**Statistical analysis** is the science of collecting, exploriing and presenting large amounts of data to discover underlying patterns, behaviours and trends.
Statistical analysis include: 
- descriptive statistics where models summarize available data to describe past events.
- Inferential statistics where models take the results of a sample and generalize and extrapolae them to a large population.
- predictive modeling in which models provide an estimate about the likelihood of a future outcome. This outcome can be a binary target(yes/no), a multinomial target(high, medium or low) or a continous target( where the event can be continous values).
- prescriptive statistics, where models quantify the effect of the future decisions. This simulates and evaluates several possible courses of action and allows companies to assess  different possible outcomes based on these actions. It is like aa wat-if type of analysis.

**Forecasting** describes an observed time series to understand the underlying causes of changes and to predict future values.
- Auto-regressive integrated moving average (ARIMA) models are forecasting models where the predictions are based on a linear combination of past values, past errors and current and past values of the other time series.
- Casual model is another type of forecasting model which forecasts time series data that are influenced by casula factors such as calendar events to describe possible seasonality.
- There are modern and complex forecasting models that incorporate time series data whose level, trend, or seasonal components vary with time. They might include hierachical segments of time series and recurrent neural networks to account for stationary and non-stationary data.

**Text analytics** is a field associated with uncovering insights from text data, susally combining the power of natural language processing, machine learning and linguistic rules.
It is used to analyze unstructured text, extract relevant information and transform it into useful business intelligence.
- sentiment analysis is a field in text analytics that determines levels of agreement from unstructured data associating the overall information as a positive, negative or neutral sentiment
- topics discovery and clustering. topics or clusters are revealed from various text documents based on the similarity that they have between them.
- Text categorization is a technique where a text analytics model labels natural language texts with relevantcategories from a predefined set. Domain experts and linguistics interactin this field to create and evaluate the categories.

**Survival Analysis** is a class of statistical methods from which the outcome variables of interest is the time until an event occurs.
The basis of the analysis is the time at risk for the event. Therefore it is not just whether the event occurred but when it occurred.

**Optimization** an optimization model searches for an optimal solution, which considers a set of pre-determined constraints and a limited set of resources.
Network analysis and network optimization involves analysis of networks(nodes and links) and analysis of network flow.

## DATA SCIENCE PRACTICAL EXPERIENCE

**Customer experience**
- enhanced customer experience
- insolvency and default prediction - monitor cutomer's expences and usage over time.
- churn prediction and prevention
- next best offer - operationalize customer insights
- target social influencers

**Revenue Optimization**
- Products bundles and marketing campaigns - improves business decision making process
- revenue leakage- identify areas with revenue leakage
- personalize products and services
- rate plans and bundles

**Network analytics**
- Network capacity planning - 
- service assurance and optimization - prevent network problems before they happen
- optimize supply chain - best routes to deliver goods
- Unstructured data- for deeper customer and service insight

**Data Monetization**
- location-based marketing
- micro segmentation
- Third party partnership
- real-time data analysis


# CH:2 DATA EXPLORATION AND PREPARATION

## INTRODUCTION TO DATA EXPLORATION
Data scientists get to know the data they will use to solve any business problem.
Tasks associated with data exploration include managing the following:

**Nonlinearity**
Data exploration can illustrate nonlinear association between the predictor variable and the outcome
Non-linearity demands exploration of different models to account for the complex relationships  between the predictor variables and the Target. Universal approximator models such as tree-based models and neural networks can be used here to mitigate these complex relationships.

**High Cardinality**
This can be very problematic in analytical models like regression or neural networks because for each level it create a parameter estimate or a weighted parameter, respectively. These levels essentially create models that are much more complex at the end of the process.

**Unstructured Data**
Using unstructured data such as textual data, network data, images, audio and video might help explain the business problem or simply hold a greater predictive power in relation to the outcome data scientists are modeling.
These types of data should be used in analytical models either to solve business problems directly or to add new features for improving model performance and interoretability

**Sparse Data**
situations where there are few events in the data nad most of the occurrrences are mmissing or incomplete.
Factorization machine models circumvent(avoid/ get around) the problem of sparse matrices by using matrix factorization.

**Outliers**
Outliers can be detected during data exploration, and data scientists must decide whether these data points are erroneous, or they depict unusual circumstances.
Sometimes finding outliers might represent a solution to a business problem. An oulier can represent an anomaly, a correct value that is very unusual and very unexpected. This outlier can highlight a potential fraud event, a failure in machinery or a default in financial operations.
There are techniques to reduce the effect of the ouliers in the predictive models and other techniques to eliminate the outliers from the training and validation processes.

**Mis-scaled Input Variables**
Data scientists should examine the scale of the predictor variables. In some modeling situations, re-scaling the variables to the same scale mighr be beneficial. 
In models like Regression and tree-based models, different scales might not be a big problem.
For complex models like neural networks, different scales might represent a bias for the way the model considers and accounts for the distinct inputs.

An imprtant goal in data exploration is to understand all the possible data sources and predictor variables that are used to solve the business problem based on analytical reports.

## INTRODUCTION TO DATA PREPARATION
Data scientists need to access all the data sources available in the company to solve the business problem.
Tasks associated with data preparation include:

**Representative Sampling**
Sampling is done for data efficiency purposes because smaller sample sizes for data preparation are usually preferred. Splitting the data into multiple samples enables data scientists to assess how well models generalize the predictions for future data.

**Event-based Sampling**
In business situations, the target that we are trying to model is rare.
A common strategy in predicting rare events is to build a model on a sample consisting of all the events and the merge to a sample of non-events.
The advantage of event-based sampling is that data scientists can obtain, on average, a model of similar predictive power with a smaller overall case count.
This sampling strategy works because the amount of information in a dataset with a categorical outcome is determined not by the total number of observations in the dataset but by the number of observations in the rarest outcome category, which is usually the number of events.
This sampling approach allows the model to capture both relationship between the inputs and the event of interest  and the inputs and nonevents.
If you have in the trianing dataset  just nonevent or atleast a majority of them, the model tends to easily capture the relationship of the inputs and the nonevents.

**Partitioning**
Analytical models tend to learn very fast and effeciently capture the relationship between the input variables and target. The problem is they learn too much. The models can capture almost perfectly the correlation between the inputs and the target, but just for the timeframe that it has been trained. The models should generalize as much as possible to account for the variability of the data in different timeframes.
When the model has high predictive accuracy for the trained period but not for future data, it is said to that the model *overfits* the data
The simplest strategy for correcting overfitting is to isolate a portion of the data for assessment or validation.
The model is fit to one part of the data, called the training dataset , and the performance is evaluated on another part of the data called the validation dataset.
When the validation data are used for comparing, selecting and modifying models, and choses=n model is assessed on the same data that was used for comparison, then the overfitting principle still applies. In this situation a test dataset should be used for final assessment.
In situations where there is a time component, the test dataset could be gathered from a different time.This would generalize the model even more as the model should be deployed in a different timeframe.

**Imputation**
standard approach for handling missing data is *complete-case analysis* where only those observations without any missing values are used in the training process. that means any observation with a missing value in any of the input variables used to train the model is eliminated from the training process. The problem is even a small missing value can cause an enormous loss of data in high dimensions.
Imputation means filling in the missing values with a reasonable value using methods such as mean, median, mode and midrange imputation. These values are based on the distribution of the predictor variables.
There are other methods to handle missing values like:
- binning the variable into groupsand having one group represent all the missing values. The predictor variable would then be treated as a categorical variable in the analysis.
- estimate the missing values based on the inputs. Data scientists would use another predictive model to estimate the missing values before estimating the target

**Replacement**
data can have erroneous values that need to be replaced. This can be done by imputating the necessary values based on other predictor variable values in the data
If the percentage of errors is too high, the data scientists can decid eto discard this variable from the analysis.

**Transformation**
Transformation can be a reasonable method to improve the fit of the model.
Data scientists can take a mathematical transformation to create a more symmetric distribution that should reduce the effect of outliers or heavy tails in the original variables' distribution.
Data scientists can also standardize variables to be in the same scale and range. 
Variable transformation can help the modelcapture nonlinear relationships between the inputs and targets.
Another transformation is to bin the input variable values and treat it as a categorical variable. This can also reduce the effect of outliers and account for nonlinear relationships between the target and the input variables.

**Feature Extraction**
Otherwise known as feature creation create new variables/features from the initial set of variables by combining the original variables. The features encapsulates the important information from multiple original input variables
In many situation feature extraction is based on domain knowledge.
Feature creation can be based on text analytics where unstructured text is converted to predictor variables. Principle component analysis (PCA) and data transformation can be used to create new features.

- *Principal component analysis* are weighted linear combinations of the predictor variables where the weights are chosen to account for the largets amount of variation in the data. Total variation in this case is the sample variance of the predictor variables. The principal components are numbered according to how much variation in the data is accounted for. Since the principal components are orthogonal(right angle) to each other, each principal component accounts for a unique portion of the variation in the data. Principal component analysis can be used for dimension reduction since usually only the first few principal components explain enough of the proportion of the total variation in the data. Example, if the data scientists have 1000 variables but the first 10 principle comopents explain 95% of the variation in the data, then only the first 10 principle components would be used in the model. It reduces the high dimensionality in the input space, the original variables and creates more informative variables for the model.

- *Text Mining* Text parsing processes textual data into a term-by-document frequency matrix. Transformations such as singular value decomposition(SVD) alter this matrix into a dtat set that is suitable for predictive modeling purposes. SVD is simply a matrix decomposition method. When it is used in a document by term matrix, a set of predictor variables is produced that contains information about what is written in the text. The set of coefficients that are produced can be used to derive concepts or topics from the document collection. Example: The document could be Call Center dialogues and the terms could be: angry, frustrated, upset, etc... The coefficient can be used to create variables that represent the strength of the term in the document. Variables that show the strength of terms associated with unhappy customers should be useful predictors in a churn model. 

- *Variable Clustering* finds groups of variables that are as correlated as possible among thmeselves and as uncorrelated as possible with variables in other clusters. This is used as a dimension reduction technique as the data scientists would choose one variable from each cluster based on subject matter knowledge. The data scientists could also choose a representative variable from each cluster based on the correlation with its own cluster and the correlation with the other clusters.

- *Autoencoder* Autoencoder data transformation is an algorithm that aims to transform the predictor variables into derived variables with the least amount of distortion. It attempts to discover structure within the data to develop a compressed representation of the original variable. The first step is *encoding* where the algorithm efficiently compresses the data. The second step is *decoding* where the algorithm tries to reconstruct the original data. **The decoder is used to train the encoder**
If the predictor variables were independent of each other, the encoding step would produce derived inputs with very limited value. If some sort of structure exists in the data such as correlated predictor variables, then the encoding step would produce derived inputs that reconstruct the predictor variables without holding on to the redundancies within the variable.
Autoencoder data transformation can be used for dimension reduction. Principal componets analysis attempts to dicover linear relationships in the data while the derived variables from the autoencoder can discover nonlinear relationships. Whereas principal components are uncorrelated, the derived variabled might have correlations since autoencoding strives for accurate reconstruction of the variables.

**Feature Selection**
also known as dimension reduction. Data scientists want to eliminate irrelevant and redundant variables without eliminating important ones. Some of the dimension reduction methods are *correlation analysis*, *regression analysis* and *variable clustering*.

When fetures are created, a common recommendation is to eliminate the original variables that were used to construct the features because the features and the original variables will probably be redundant. However, another point of view is to keep all the variables for the dimension reduction techniques and see which ones survive to the final model.

Redundancy among predictor variables is an unsupervised concept since it does not involve the target variable. The relevancy of a variable considers the relationship between the predictor and target variable. In high dimensional data sets, identifying irrelevant variables is more difficult that identifying redundant variables. A good strategy is to first reduce redundancy and then tackle irrelevancy in a lower dimension space. 
A rendundant variable does not give any new information that was not already explained by other variables, eg, knowing the value of household income usually indicates home value, as one increases the other one increases.
An irrelevant variable does not provide information about the target.

**MODEL SELECTION**
once a reasonable number of variables have been selected, data scientists usually have  several choices of models to choose based on their complexity. A common pitfall is to overfit the data in which the data model is too complex. An overly complex model might be too sensitive to the noise in the data and not generalize well to new data. Using too simple model can lead to underfitting where the true features are ignored.

**Model Generalization**
Typically, model performance follows a straight forward trend. AS the complexity of the model increases and more terms are addded to the model, the fit on the training data set generally improves as well, Some of this increase is attributable to the model capturing relevant trends in the data, sometimes very specific correlation between inputs and the target. However, some of the increase might be due to overfitting the training data as the model might be reacting to random noise. That will work well for the trianing data set but not for the validating data set or even the test data set and probably not for future data, therefore, the model fit on the validation  data for the models of varying is also examined.

The typical behaviour of the model fit on the validation data is an increase in model fit as the complexity increaes, followed by a plateau, followed by a decline in performance. This decline is due to overfitting. Consequently, a common recommendation is to select the simplest model with the best performance. That means select select the model with the best performance under the validation data but the simplest model possible. If the assessment error plot shows a plateau in the error for the validation data set, then select the simplest model at the beginning of the plateau. Simpler models tend to generalize better. Complex models tend to be affected by change in the input data because their structure is more complex.

**Bias-Variance Tradeoff**
The goal of a data scientist is to fit a model with low bias and low variance. 
- *bias* is the difference between the average prediction of the model and the correct values that we are trying to predict.A model with high bias misses the important relationships between the predictor variables and the target. This is an example of underfitting. 
- *variance* is the variablity of model prediction for a given data point. A model with high variance models the random noise in the data. This is an example of overfitting.
The Bias-Variance tradeoff is where the data scientists choose a model that accurately depicts the relationships in the data, but also chooses a model that generalizes well to new data.


# CHAPTER 3: SUPERVISED MODELS - Statistical Approach
Supervised refers to observations where the target is known
based on the known events, a model is trained to capture the relationship between inputs and target

## Classification and Estimation
Classification and estimation are commmon tyoes of predictive models. 
Classification assumes that the target is a class variable. The target can be a binary class, 0 or 1, yes or no, or it can be a multinomial class 1,2,3,4,5 or low, medium, high. eg(is this business event a fradulent transaction(yes or no)?)
Estimation assumes that the target is a continous number. The taret can take any value in the range of negative infinity to positive infinity.

Both predictive models require the following:
- Observations/Cases/Instances: a real case comprising of attributes that describe the observation
- Inputs/attributes/variables: the measure of the observation or its attributes. It can be demographic information about the customer such as name, salary, age and marital status.
- Target/class/label: a tag or label for each observation. eg default or no default.

Statistical models map the set of input variables to the target. The model tries to create a concise representation of the inputs and the target.
An example: our input is: charachteristics of a customer such as age, gender, how long they have been in the DB, marital status...etc. our target is purchasing company product. By going through past event of how the target has related with the product we get to know the type of person based on their charachteristics who is likely to buy this product. From this the company will target perople with these specific charachteristics to purchase the product. of course, some targeted customers will eventually not purchase the product and some customers not targeted will purchase the product. This is the *intrinsic error* associated with the model. Customers with similar charachteristics behave very differently. Errors are associated with all models.

All supervised models behave similarly, using the past known events to estimate the relationships between predictor variables and the target. If a good model is found the model can be used to predict future events.

Because the purpose of the supervised model is to predict the unknown values of the target, the main goal of the supervised model is to generalize to new data or future cases. **Generalization** means the ability to predict the outcome on new cases when the target is unknown. That is the reason we use different data sets to train the model(capture the relationships between the inputs and the target) and validate the model(generalize the model to make good predictions on the new data).

Generalization is involved in model assessment. The model is fit to the training data set, and the performance is evaluated on the validation data set by comparing the predicted values to the observed values of the target. Since the target value is known, the assessment is straightforward, comparing the values the model predicts for the target and the values observed for the target.

This chapter discusses 3 distinct types of models:
- Linear Regression: used to predict continous targets. one assumption of the linear regression model is that there is a linear relationship between the input and the target
- Logistic Regression: used to predict discrete targets such as binary, ordinal and nominal outcomes. one assumption of the logistics regression is that there is linear relaationships between the inputs and the logits.
- Decision tree: used to predict both continous and categorical targets. There is no assumption for the relationship between input and target. Decision trees are universal approximators because theoritically they can capture any type of relatioship between inputs and targets.

## Linear Regression
The relationship between the target and the input variables can be charachterized by the equation: y1=Bo + B1X1 + E1, i = 1...,n
where: 
- y1: is the target variable, 
- X1: is the input variable, 
- Bo: is the intercept parameter, which corresponds to the value of the target variable whe the predictor is 0.
- B1: is the estimate(slope) parameter, which corresponds to the magnitude of the change in the target variable given a one-unit change in the input variable.
- E1: is the error term representing devistions of y1 about Bo+B1X1

Estimates of the unknown parameters Bo and B1 are obtained interactively by the method of ordinary least squares. This method provides the estimates by determining the line that minimizes the sum of the squared vertical distances between the observations and the fitted line. in other words, the fitted or regression line is as close as possible to all the data points.

The estimation for the target variable is formed from simple linear combination of the inputs. The prediction estimates can be viewed as a linear approximation to the expected value of a target conditioned on the observed input values.
Lineear regression is quite simple to implement since it is only a linear equation. The coefficients from the model are easily interpretable as the effect on the target given a one-unit increase in the predictor variable controlling for the other predictor variables. In most cases, the results of the model can be obtained quickly, and new observations can be scored quickly. Predictions on new cases can be easily obtained by plugging in the new values of the predictor variables.

The *drawback* of linear regression are:
- the models are limited to normally distributed residuals, which are the observed values minus the predictor values. If the residuals show a non-normal distribution such as skewed distribution, then a generalized linear regression model might be helpful.
- High degree of collinearity among the predictors can cause model instability. The instability of the coefficients might not lead to a simple model that identifies the predictor variables that are the best predictors of the target. Therefore, Data scientists should reduce redundacy first before using linear regression models.
If there are non-linear relationships between the predictor variables and the target, data scientists will have to add higher order terms to properly model these relationships or perform variable trasformations. Finding the appropriate higher order term or variable transformation can be time-consuming and cumbersome.

Linear regression models can be affected by outliers that can chage the model results and lead to poor predictive performance.

## Logistic Regression
Closely related to linear regression. In Logistic regression, the expected value of the target is transformed by a link function to restrict its values to the unit interval. In this way, model oredictions can be viewed as primary outcome probabilities between 0 and 1. A linear combination of the inputs generates a logit score, or the log of the odds of the primary outcome, in contast to linear regression which estimates the value of the target.
The range of logit scores is from negative infinity to positive infinity.
For binary predictions, any monotonic function that maps the unit intervals to the real number line can be considered as a link. The logit link function is one of the most common. Its popularity is due, in part, to the interpretability of the model.
The continous logit scores/logit of *P*, is given by: log of the odds(log of the probability of the event), divided by, the probability of non-event. This logit transformation transforms the probability scale to the real line of negative infinity to positive infinity. Therefore, the logit can be modeled with a linea combinantion since linear combination can take on any value.

The Logistic model is particularly easy to interpret because each predictor variable affects the logit linearly. The coefficients are the slopes. Exponentiating each parameter estimate gives the odds ratios, which compares the odds of the event in one group to the odds of the event in another group.

The predictions from logistic regression can be rankings, decisions or estimates. Analysts can rank the posterior probabilities to assess observations to decide what action to take.
The parameter estimates in a logistic regression are commonly obtained by the method of maximum likelihood estimation. These estimates can be used in the logit and logistic equation to obtain predictions.
If the target variable is in 2 categories, the appropriate logistic model is *binary logistic regression*
If there are more than 2 categories within the target variable:
- If the target variable is nominal, the appropriate model is nominal logistic regression
- If the target variable is ordinal(ranking), the appropriate model is ordinal logistic regression.
The binary logistic regression model assumes that the logit of the posterior probability is a linear combination of the predictor variable.

Strengths of Logistic regression model are:
- The model is simple to implement since its only a linear equation
- The coefficients from the model are easily interpretable as the effect on the logit gives a one-unit increase in the predictor variable controllong for the other predictor variables.
- The results of the model can be obtained quickly.
- new observations can be scored quickly.

Drawbacks of Logistic regression model are similar to linear regression drawbacks.

## Decision Tree
Decision trees are statistical models designed for supervised prediction problems. Supervised prediction encompasses predictive modeling, pattern recognition, discriminant analysis, multivariate function estimation, and supervised machine learning. A decision tree includes the following components:
- an internal node is a test on an attribute
- a branch represents an outcome of the test, such as color=purple.
-  a leaf node represents a class label or class label distribution.
- at each node, one attribute is chosen to split the training data into distinct classes as much as possible.
- a new instance is classified by following a matching path to a leaf node.

The model can be represented in a tree-like structure.
A decision tree is read from top down starting from the root node. Each internal node represents a split based on the values of one of the inputs The inputs can appear in any number of splits throughout the tree. Cases move down the branch that contains its input value.
In a binary tree with interval inputs, each internal node is a simple inequality. A case moves left if the inequality is true and right otherwise. The terminal nodes of the tree are called leaves, they represent the predicted target. All cases reaching a leaf are given the same predicted value. The leaves give the predicted class as well as the probability of class membership.
Decision trees can have multi-way splits where the values of the inputs are partitioned into ranges.

When the target is categorical, the model is called a classification tree. **A classification tree** defines several multivariate step functions. Each function corresponds to the posterior probability of a target class. when the target is continous the model is called a **regression tree**
Cases are scored using prediction rules, these rules define the regions of the input space purer with regard to the target response value.

The strengths of decision trees are that they are easy to implement and they are very intuitive. The results of the model are easy to explain to non-technical personnel. Decision trees can be fit very quickly and can score new customers very easily. It can handle non-linear relationships between target and the predictor variables without specifying the relationship in the model unlike logistic and linear regression.
Missing values are handled because they are part of the prediction rules.
Decision trees are also robust to outliers in the predictor variable values and can discover interactions. An interaction occurs when the relationship between the target and the predictor variable changes by the level of another predictor variable e.g if the relationship between the target and income us different for males compared to females, decision trees would discover it.

Decision trees confront the curse of dimensionality by ignoring irrelevant predictor variables. however they have no built-in method for ignoring redundant predictors. This can be a problem in deployment in that decision trees might arbitrarily select from a set of correlated predictor variables. It is recommended that data scientists reduce redundancy before fitting the decision tree.

A drawback: Decision trees are very unstable models, any minor changes in the training data set can cause substantial changes in the structure of the tree. The overall performance, or accuracy, can remain the same but the structure can be quite different. The common method to mitigate the problem of instability is to create an **ensemble of trees**.


# CHAPTER 4: SUPERVISED MODELS- MACHINE LEARNING Approach
3 types of diverse models:
- **Random forest** used to predict nominal targets or estimate continous values. There are no assumptions for the relationships between the inputs and the target. The random forest model is based on multiple independent decision tree models.
- **Gradient boosting** used to predict nominal targets or estimate continous values. There are no assumptions for the relationship between the input and the target. Gradient boosting is based on a sequence of decision tree models.
- **Neural networks** used to predict nominal targets or estimate continous values. This model is based on the linear combination of nonlinear multiple regression. The combination of multiple inputs and hidden neurons allows the model to account for nonlinearities between the inputs and the targets.
All 3 models are universal approximators because theoritically, they can capture any type of relationship between inputs and the target.

## ENSEMBLE OF TREES
Decision trees are particularly good models, easy to implement, fast to train and easy to interpret, but they are unstable. The instability results from the considerable number of univariate splits and fragmentation of the data. At each split there are typically many splits on the same predictor variable or different predictor variables that give similar performance. eg: suppose age splits at 45 since it is the most significant split with the predictor variable and the target. however, other splits at 37 or 47 might also be significant. A slight change in data can easily result in an effect that can cascade and create a different tree. A change in the input data can result in a split at 38 years or even result in another input variable being more significant to the target such as income. Then. instead of splitting age, the decision tree will start splitting input space based on income. The final tree structure will be quite different, and therefore, the set of rules and thresholds will also be different.

**ensemble tree** is a combination of multiple models. The combination can be formed in these ways:
- voting on the calissifications
- using weighted voting, where some models have more weight
- averaging(weighted or unweighted) the predicted values
There are 2 types of ensemble trees: *Random forest* and *gradient boosting models*

## Random Forest
The training data for each decision tree is sampled with replacement from all observations that were originally in the training data set. *sampling with replacement* means that the observation that was sampled is returned to the training data set before the next observation is sampled, and therefore each observation has a chance of being selected for the sample again. Furthermore, the predictor variables considered for splitting any given decision tree are randomly selected from all available predictor variables. Different algorithms randomly select a specific number of predictor variables. e.g: at eaxh split point, only a subset of predictors equal to the square root of the total number of predictors might be available. Therefore, each decision tree is created on a sample of predictor variables and from a sample of the observations. Repeating this process many times leads to greater diversity in the trees. The final model is a combination of the indiviadual decision trees where the predicted values are averaged.
Forest models usually have improved predictive accuracy over single decision trees because of variance reduction. If individual decision trees have low bias but high variance, then averaging them decreases the variance.

Random forest is based on a concept called **bagging** *bootstrap aggregation*. It is the original perturb and combine method developed by Breiman in 1996. The main idea of the perturb and the combine method is to take the disadvantages of decision trees in terms of instability and turn it into a major advantage in terms of ronustness and generalization. These are the main steps:
- **Draw K bootstrap samples** A bootstrap sample is a random sample of size x drawn from the empirical distribution of sample size x. That is, the training data are resampled with replacement. Some of the cases are left out of the sample, and some are represented more than once. (creating multiple data sets based on the original data set replicating the rows.)
- **Build a tree on each bootstrap sample** (build a tree on each sampled data set) Large trees with low bias are ideal.
- **Vote/Average** for classification problems, take the mean of the posterior probabilities or the plural vote of the predicted class. Averaging the posterior probabilities gives a slightly better performance than voting. Take a mean of the predicted values for regression.

## Gradient Boosting
This is a weighted linear combination of individual decision trees. The algorithm starts with an initial decision tree and generates the residuals(errors). In the next step, the target is the residuals from the previous decision tree. At each step the accuracy of the tree is computed, and successive trees are adjusted to accommodate orevious inaccuracies. Therefore the gradient boosting algorithm fits a sequence of trees based on the residuals from the previous trees. The final model also has predicted the values average over the decision trees.
The steps are as follows:
- 1 Build a model and make predictions (makes model 1)
- 2 Calculate the error and set this as target. (difference between actual data and the predicted data)
- 3 Build model on the errors and make predictions. (using the errors as the target) the prediction will be error values.
- 4 Predicted error is added to model 1 (makes model 2)
- 5 repeat step 2 to 4

Major difference between random forest and gradient boosting is in the way the ensemble decision tree is created. In forest, eacg decision tree is created independently while in gradient boosing, the set of decion trees is created in a sequence. This difference can allow random forest to be trained faster and gradient boosting to be more accurate. Random forest can better generalize and gradient boosting easier to overfit.

The ensemble tree models have low bias and variance, they can handle non-linear relationships with the target, they can handle missing values and are robust to outliers
The weakness of ensemble models are that the simple interpretation of a single decision tree is lost. Forest and gradient boosting models are not simple and interpretable. They also require far more computer resources compared to the single decision tree.

## Neural Networks
If the relationships between the input variables and the target is nonlinear but it is possible to specify a hypothetical relationship between them, a **parametric nonlinear** regression model can be built. When it is not practical to specify the hypothetical relationsip, a **nonparametric regression model** is required.
Nonlinear regression models are more difficult to estimate than linear models. Data scientists must specify the full nonlinear regression expression to be modeled and an optimization method to efficiently search for the parameters. Initial parameter estimates also need to be provided and are critical to the optimization process. Another option is a **nonparametric regression model** that has no funtional form/parameters.
Traditional nonlinear modeling techniques are more difficult to define as the number of inputs increase. It is uncommon to see parametric nonlinear regression models with more than a few inputs, because deriving suitable functional form becomes more difficult as the number of input increases. Higher-dimensional input spaces are also a challenge for nonparametric regression models.
Neural networks were developed to overcome these challenges. Although neural networks are parametric nonlinear models, they are nonparametric in one way: neural networks do not require the functional form to be specified. This enables data scientists to construct models when the relationships between the inputs and the target are unknown.
However, like other parametric nonlinear models, neural networks do require the use of an optimization process with initial parameter estimates, unlike nonlinear parameter regression models and nonparametric models, neural networks perform well in sparse high-dimesional spaces.
Both regressions and neural networks have similar components but with different names. inercept estimate is bias estimate, parameter estimate is weight estimate in neural networks
The prediction formula used to predict new cases is like a regression model but with a very flexible addition. This addition enables a trianed neural network to model any association between the input variable and the target. Flexibility comes at a price because of the lack of a built-in method for selecting useful inputs.
The model is arranged in layers. 
- The first layer is Input layer, consisting of the input variables. 
- The second layer is the hidden layer, consisting of hidden units/neurons. It is possible to have multiple hidden layers with multiple hidden units in each hidden layer.
- Target layer, third layer consisting of the response/target/output.
Neural networks predict cases using a mathematical equation involving the values of the input variables. The inputs are weighted and linearly combined through hidden units in the hidden layer. The hidden units include a link function **activation function** to scale outputs. The result is compared to the observed value in the target layer, the residual is computed and the weights are re-estimated.
The output from a neural network with one hidden layer is a weighted linear combination of the mathematical function generated by the hidden units. The weights and biases give these functions their flexibilities. Changing the orientation and steepness of these functions and then combining them, enables the neural network to fit any target. After the prediction formula is generated, obtaining a prediction is simply a matter of plugging the predictor variable values into the hidden input expressions. Data scientists obtain the prediction estimates using the appropriate link function in the prediction equation.
Neural networks are universal approximators that can model any relationship between the predictor variable and the target. They are also robust to outliers.

Number of hidden unit required is subjective. If the neural network has too many units, it will model random variations as well as the desired pattern. If it has few hidden units, it will fail to capture the underlying goal. Specifying the correct number requires some trial and error. Data scientists can specify the number by using the **goodness-of-fit statistic** on the validation data set. eg: fir several models with different number of hidden units and choose the model with the lowest goodness-of-fit statistic on the validation set.
Finding reasonable values for the weights is done by leasr squares estimation for interval-valued targets and maximum likelihood for categorical targets. The search for the weights involves an optimization process, which applies an error function/objective function to update the previvous wheights at each iteration. When the target variable is binary, the main neural network regression equation receives the same logit link function featured in logostic regression.
Complex optimization algorithms are a critical part of neural network modeling. The method of stopped training starts with a randomly generated set of initial weights. The trianing proceeds by updating the weights in a manner that improves the value of the selected fit statistic. Each iteration in the process is treated as a seperate model. The iteration with the best value is chosen as the final model. To avoid overfitting **weight decay** is applied during the search for the right set of weights. It is observed that every time the shape of the activation function gets too steep, the neural network overfits, this is caused by the weights getting too large. The weight decay method penalizes the weights when they grow too large by applyig regularization. Tere are 2 tyoes of regularization: 
- Lasso or L1 - penalizes the absolute value of the weight.
- Ridge or L2 - penalizes the square of the weight
Another important parameter is learning rate, which controls how quickly training occurs by updating the set of weights in each iteration.
Neural networks take a lot of time when fitting and training the model especially as the number of hidden units and layers increase. They sometimes do not perform as well as simpler models such as linear regression. This situation occurs when the signal to noise ratio is low. When there is a strong pattern(signal) relative to the variation(noise) the signal to noise ratio is high. In this situation, neural networks outperform simpler models.

# CHAPTER 5: ADVANCED TOPICS IN SUPERVISED MODELS
# Advance Machine Learning Models and Methods
This chapter covers 2 advanced machine learning models and 2 advanced methods to train supervised models. 
The machine learning models are:
- **Support vector machine** This model is implemented SAS (statistical analysis system) can currently handle binary and continuous targets.
- **Faxtorization machine** This model is an extension of the linear model and it can handle continuous targets.
The advance methods are:
- **Ensemble models** This model combines several models to predict the target value. The taget can be continuous or discrete.
- **Two-Stage Models** This model allows the modeling of a discrete and continuous target. The model first predicts the binary response and if the prediction is 1, then the model predicts the continous response.

## Support Vector Machines
SVM is a robust model used to classify categorical or continous targets. Like neural networks, these models tend to be black boxes, but they are very flexible. SVM automatically discover any relationship between the input and the target. Data scientists do not need to specify the functional form, or the relationship between the inputs and the target before fitting the model.
SVM makes decision predictions instead of ranks or estimates. In that way, it seperates the outcome of a binary target into 2 classes. It can be used for regression tasks as well. There are many classification rules/regression lines that can be used to seperate the 2 classes, if the data is linearly seperable, there are limitless number of solutions/lines to seperate the 2 classes or any case of binary target. 
Given 2 input variables( within x and y axis(2D)) the SVM  is a **line** (y=mx+c). 
Given 3 input variables (x y and z axis (3D)) the support vector is a **plane**.
With more than 3 variables, the support vector is a **hyperplane**
For mathematical convenience, the binary target is defined by +1 and -1, rather than the usual 1 and 0 because the linear seperator equals 0, classification is determined by a point falling on the positive or negative side of the line.

SVM gets more complicated when the problem is not linearly seperable.
H={<w,x> +b=0}
where w is the mechanism that affects the slope of H(the optimal line that correctly classifies the observation).
The bias parameter b is the measure of offset of the seperating line from the origin, or the plane in 3D or hyperplane in higher dimension.
The quantity <w,x> is the dot product between the vectors w and x. A **dot product** is a way to multiply vectors that result in a scalar, or a singular number, as the answer. It is an element-by-element multiplication and then a sum across the products. SVM selects values for w and b that define the iptimal line that correctly classifies the cases.

How are values w and b chosen? SVM try to find the decision boundary that maximixes the margin. Margins are the parpendicular distance between the line H and the carrying vectors. The SVM is trying to find the hyperplane that maximizes the margin with the condition that both classes are correctly classified. The properties of the hyperplane that maximizes the margin are described by the support vector also called carrying vectors. These are the points closest to the hyperplane and they determine the location of the hyperplane (H). Because the hyperplane depends only on data points and not predictor variables, the curse of dimensionality is minimized.

In most realistic scenarios, not oly are the data not linearly seperable, but a classifier defined by the hyperplane would make too many mistakes to be a viable solution. In that case, one solution would be to apply a polynomial transformation so that the input data would be projected into higher dimensions and then find the maximum margin hyperplane in this higher dimensions. It might be the case that it's not possible to find a hyperplane to linearly seperate the the classes considering a particular input data, but we could find a hyperplane in a higher dimesion that splits the input data when it is set in a higher dimensionality.
The dot product to transform the support vector in a higher dimensionality, such as polynomials, can be extremely computationally intensive. to mitigate this, the algorithm to calculate the support vectors uses a kernel function, such as a polynomial kernel of degree 2 or 3, to compute the dot product of the 2 products that is much less computationaly intensive. This is known as a **kernel trick**
 One of the strengths of SVM is that data scientists do not need to specify the functional form or know the type of relationship between inputs and targets. 
 SVM  are robust to outliers in the input space. They sre effective classifiers when the data points are seperable in the input space.
 SVM are less affected by the curse of dimensionality because it is less affected by the number of predictor variables compared to other models. The hyperplane only depends on data points and not predictor variables.

One of the weaknesses of SVM is that the model itself is a black box with no interpretable parameters estimates or rules and thresholds. SVM work very similarly to neural networks and tree-based ensemble models like random-forest and gradient boosting when the subject id interpreting results. In regulated markets where the relationships between the predictor variables and the target are important to understand the outcome, and mostly to explain the results, SVM would have limited usefulness. In neural networks models, a surrogate model such as a regression or a decision tree, can be used to explain the outcomes from a SVM. However, by using a surrogate model the data scientist are not explaining the models but instead they are interpreting the results in terms of the business actions. Some regulators may not accept this approach.


## Factorization Machines
There is a class of business applications that involves estimating how users would rate some items. Very often companies do not have much information about the users and items. The main goal in this model is to evaluate the relationship between users and items.
Factorization machines is a powerful tool for modeling high fimensional and sparse data. A sparse matrix is the matrix of users and items. The main goal of factorization machines is to predict the missing entries in the matrix.
FM are used in recommender systems where the aim is to predict user ratings on items. There are 2 major types of recommender systems: One type relies on the on the **content filtering** method. Content filtering assumes that a set of information about the items is available for the model estimation. This set of information is often refferef to as side informtion, and it describes the items that will be recommended. By having side information available, the content filtering has no problem in estimating the rating for a new item.
The second type is based on **collaborative filtering**. This method does not require additional information about he items. The information needed is the matrix containing the set of users, set of items and the ratings of users to items. The combination of all users and items create a large, sparse matrix containing lots of missing ratings. The collaborative method works well in  estimating ratings for all combinations of users and items, The problem with it, is the inability to estimte added items that have no rating.

**Factorization machines** estimate the ratings by summing up the average rating over all users and items, the average rating given by a user, the average rating given to an item, and a pairwise interaction term that acounts for the affinity between a user and an item. This affinity is modeld as the inner product between 2 vectors of features: one for the user and another for the item. These features are collectively known as factors.
Matrix factorization techniques are effective because they enable data scientists to discover the latent features underlying the interactions between users and items. The basic idea is to discover 22 or more matrices that when multiplies, they return the original matrix.


## Ensemble Models
create new models by combining the predictions from multiple models consisting of different modelling techniques or different modelling structures. The combined model is the used to score data. For binary target, data scientists can take the mean of the posterior probabilities as the final predicted probability. Another approach is to derive a decision and see how many models predict a 1 versus a 0, then tke a plurality vote of the predicted class. For continous target, ususal approach is to make the final estimation as the mean of the estimations for each model in the ensemble structure.
The common advantagwe of ensemble models is tha the combined model is usually better than the individual models that compose it. Better in terms of generalization because the final model accounts for relationships discovered from distict models. The combination of these models can better account for variablity of future data. It is also better in terms of accuracy. The overall estimation  can present less errors than individual models. Dat scientists should always compare the model performance of the ensemble and individual models and evaluate when to deploy which model among the 2.
Ensemble models have low bias and low vvariance and are more robust to overfitting.
The ensemble model can generalize better to future data and is be more accurate.

Ensemble models lack intepretable parameter estimates. Therefore if understanding the relationship between the input and target is a priority then the model is not useful. The model might also be slow since many models are in production.
If the relationship between input and trget are captured by a single model, ensemble approach offers no benefit
Ensemble models are more computationally expensive since several models are being fitted and the results are averaged.
The model cannot be interpreted.

## Two-Stage Models
This model enables an data scientists to model a class target and the use the predicted probability, or the decision , to split the observations and feed them into another model, normally to train a model with a continous target. The 2 models can use different algorithms and predictor variables. Therefore, the first model classifies a binary target. Based on the decision of the first model(0 or 1), another model is trained to estimate a continous value.  example: model the probability of unmanageable debt, and if the predicted class is 1, then model the amount of unmanageable debt.
If predicted probability is accurately estimated in the first stage, then there might be an increase in the accuracy of the predicted amounts in the second stage.
Data scienists need to correctly specify models in either stage because this impacts the accuracy of the predictions and therefore the estimations. The 2 stages also need to be correlated.