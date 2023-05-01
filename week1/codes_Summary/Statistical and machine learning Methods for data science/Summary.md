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
