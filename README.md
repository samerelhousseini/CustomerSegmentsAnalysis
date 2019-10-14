# Machine Learning Engineer Nanodegree
## Unsupervised Learning
## Project: Creating Customer Segments

Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

## Getting Started

In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.

The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.

Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    

```

    Wholesale customers dataset has 440 samples with 6 features each.


## Data Exploration
In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.

Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.


```python
# Display a description of the dataset
display(data.describe())

import matplotlib.pyplot as plt
import seaborn as sb
sb.set()

g = sb.PairGrid(np.log(data))
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)

```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12000.297727</td>
      <td>5796.265909</td>
      <td>7951.277273</td>
      <td>3071.931818</td>
      <td>2881.493182</td>
      <td>1524.870455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12647.328865</td>
      <td>7380.377175</td>
      <td>9503.162829</td>
      <td>4854.673333</td>
      <td>4767.854448</td>
      <td>2820.105937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3127.750000</td>
      <td>1533.000000</td>
      <td>2153.000000</td>
      <td>742.250000</td>
      <td>256.750000</td>
      <td>408.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8504.000000</td>
      <td>3627.000000</td>
      <td>4755.500000</td>
      <td>1526.000000</td>
      <td>816.500000</td>
      <td>965.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16933.750000</td>
      <td>7190.250000</td>
      <td>10655.750000</td>
      <td>3554.250000</td>
      <td>3922.000000</td>
      <td>1820.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>112151.000000</td>
      <td>73498.000000</td>
      <td>92780.000000</td>
      <td>60869.000000</td>
      <td>40827.000000</td>
      <td>47943.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](imgs/outpput_5_1.png)


### Implementation: Selecting Samples
To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.


```python
# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [71, 85, 125]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
#samples["total"] = samples.sum(axis=1)

print "Chosen samples of wholesale customers dataset:"
display(samples)

#mean	12000.297727	5796.265909 	7951.277273 	3071.931818 	2881.493182 	1524.870455
#std	12647.328865	7380.377175 	9503.162829 	4854.673333 	4767.854448 	2820.105937

sb.heatmap((samples-data.mean())/data.std(ddof=0), annot=True, cbar=False, square=True)

```

    Chosen samples of wholesale customers dataset:



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18291</td>
      <td>1266</td>
      <td>21042</td>
      <td>5373</td>
      <td>4173</td>
      <td>14472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16117</td>
      <td>46197</td>
      <td>92780</td>
      <td>1026</td>
      <td>40827</td>
      <td>2944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76237</td>
      <td>3473</td>
      <td>7102</td>
      <td>16538</td>
      <td>778</td>
      <td>918</td>
    </tr>
  </tbody>
</table>
</div>





    <matplotlib.axes._subplots.AxesSubplot at 0x11064d190>




![png](imgs/outpput_7_3.png)


### Question 1
Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
*What kind of establishment (customer) could each of the three samples you've chosen represent?*  
**Hint:** Examples of establishments include places like markets, cafes, and retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant.

**Answer:**
The answers below are based on the observations based on the heatmap above, which indicates the spending in relation to the respective means of each category: 

* The first establishment (index 71) might probably be a supermarket or a deli restaurant. This establishment buys lots of delicatessen goods (+4.6 std deviations on top of the Deli mean), along with fresh goods and groceries. 

* The second establishment (index 85) might probably be a grocery store or a supermarket. They buy lots of milk (+5.5 std devs) and groceries (+8.9 std devs), and lots of detergents and paper (+8 std devs).

* The third establishment (index 125) might probably be a restaurant or a coffee shop (who might use non-fresh or frozen ingredients). They buy lots of fresh (+5.1 std devs) and frozen goods (+2.8 std devs). The fresh goods might indicate fast-moving items, while the frozen might indicate the need to store some things for a longer term. 


### Implementation: Feature Relevance
One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.

In the code block below, you will need to implement the following:
 - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
 - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
   - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
 - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
 - Report the prediction score of the testing set using the regressor's `score` function.


```python
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
feature_name = 'Detergents_Paper'
new_data = data.copy()
rm_feature = np.array(new_data[feature_name])
new_features = new_data.drop(feature_name, axis = 1)

# TODO: Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_features, rm_feature, test_size = 0.25, random_state=0)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test) 
print("R^2 Score: " + str(score))

print(y_test[:5])
print(prediction[:5])

plt.scatter(y_test, prediction)
plt.show()
```

    R^2 Score: 0.728655181254
    [ 3381 12218   918 12408   204]
    [ 2730.  7883.   118.  7558.   550.]



![png](imgs/outpput_11_1.png)


### Question 2
*Which feature did you attempt to predict? What was the reported prediction score? Is this feature necessary for identifying customers' spending habits?*  
**Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data.

**Answer:**
I tried predicting the Detergents and Paper feature, and the R^2 score is around 0.72, which suggests a good deal of correlation between the purchases of other categories and this feature. 
This might make sense, since establishments like grocery stores or supermarkets purchase cleaning and paper products in bigger quantities than restaurants, and so this feature correlates closely with milk and groceries. 


Is this feature necessary for identifying customers' spending habits? I would argue that it is not, since it can be predicted with a relatively high R^2 score given the rest of the features. However, R^2 score is not that high (it's not in the 90s%), and so there is some information loss that would happen if we remove it. So it is not necessary to keep it, but better to. 

### Visualize Feature Distributions
To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.


```python
# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```

    /anaconda/envs/boston_housing/lib/python2.7/site-packages/ipykernel_launcher.py:2: FutureWarning: pandas.scatter_matrix is deprecated. Use pandas.plotting.scatter_matrix instead
      



![png](imgs/outpput_15_1.png)


### Question 3
*Are there any pairs of features which exhibit some degree of correlation? Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? How is the data for those features distributed?*  
**Hint:** Is the data normally distributed? Where do most of the data points lie? 

**Answer:**

Yes, the Detergents and Paper feature exhibits clear correlation with the Grocery and Milk features, and to a lesser degree, with the Frozen and Delicatessen features. The relationship with those features looks linear. This confirms our earlier suspicion about the relevance of this feautre, and it doesn't like it would be a useful feature to predict a specific customer. 

For the shape of the distributions of the 3 mentioned features above, on a first look, the features look like they have long-tail or positively skewed distributions. However, I guess that if we remove the outliers from the data, then they will look like normally distributed. 

## Data Preprocessing
In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

### Implementation: Feature Scaling
If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.

In the code block below, you will need to implement the following:
 - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
 - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.


```python
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```

    /anaconda/envs/boston_housing/lib/python2.7/site-packages/ipykernel_launcher.py:8: FutureWarning: pandas.scatter_matrix is deprecated. Use pandas.plotting.scatter_matrix instead
      



![png](imgs/outpput_20_1.png)


### Observation
After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).

Run the code below to see how the sample data has changed after having the natural logarithm applied to it.


```python
# Display the log-transformed sample data
display(log_samples)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.814164</td>
      <td>7.143618</td>
      <td>9.954276</td>
      <td>8.589142</td>
      <td>8.336390</td>
      <td>9.579971</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.687630</td>
      <td>10.740670</td>
      <td>11.437986</td>
      <td>6.933423</td>
      <td>10.617099</td>
      <td>7.987524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.241602</td>
      <td>8.152774</td>
      <td>8.868132</td>
      <td>9.713416</td>
      <td>6.656727</td>
      <td>6.822197</td>
    </tr>
  </tbody>
</table>
</div>


### Implementation: Outlier Detection
Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.

In the code block below, you will need to implement the following:
 - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
 - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
 - Assign the calculation of an outlier step for the given feature to `step`.
 - Optionally remove data points from the dataset by adding indices to the `outliers` list.

**NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
Once you have performed this implementation, the dataset will be stored in the variable `good_data`.


```python
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 =  np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    f_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(f_outliers)
    outliers += list(f_outliers.index)

outliers = outliers




```

    Data points considered outliers for the feature 'Fresh':



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.389072</td>
      <td>9.163249</td>
      <td>9.575192</td>
      <td>5.645447</td>
      <td>8.964184</td>
      <td>5.049856</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1.098612</td>
      <td>7.979339</td>
      <td>8.740657</td>
      <td>6.086775</td>
      <td>5.407172</td>
      <td>6.563856</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.135494</td>
      <td>7.869402</td>
      <td>9.001839</td>
      <td>4.976734</td>
      <td>8.262043</td>
      <td>5.379897</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>171</th>
      <td>5.298317</td>
      <td>10.160530</td>
      <td>9.894245</td>
      <td>6.478510</td>
      <td>9.079434</td>
      <td>8.740337</td>
    </tr>
    <tr>
      <th>193</th>
      <td>5.192957</td>
      <td>8.156223</td>
      <td>9.917982</td>
      <td>6.865891</td>
      <td>8.633731</td>
      <td>6.501290</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2.890372</td>
      <td>8.923191</td>
      <td>9.629380</td>
      <td>7.158514</td>
      <td>8.475746</td>
      <td>8.759669</td>
    </tr>
    <tr>
      <th>304</th>
      <td>5.081404</td>
      <td>8.917311</td>
      <td>10.117510</td>
      <td>6.424869</td>
      <td>9.374413</td>
      <td>7.787382</td>
    </tr>
    <tr>
      <th>305</th>
      <td>5.493061</td>
      <td>9.468001</td>
      <td>9.088399</td>
      <td>6.683361</td>
      <td>8.271037</td>
      <td>5.351858</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1.098612</td>
      <td>5.808142</td>
      <td>8.856661</td>
      <td>9.655090</td>
      <td>2.708050</td>
      <td>6.309918</td>
    </tr>
    <tr>
      <th>353</th>
      <td>4.762174</td>
      <td>8.742574</td>
      <td>9.961898</td>
      <td>5.429346</td>
      <td>9.069007</td>
      <td>7.013016</td>
    </tr>
    <tr>
      <th>355</th>
      <td>5.247024</td>
      <td>6.588926</td>
      <td>7.606885</td>
      <td>5.501258</td>
      <td>5.214936</td>
      <td>4.844187</td>
    </tr>
    <tr>
      <th>357</th>
      <td>3.610918</td>
      <td>7.150701</td>
      <td>10.011086</td>
      <td>4.919981</td>
      <td>8.816853</td>
      <td>4.700480</td>
    </tr>
    <tr>
      <th>412</th>
      <td>4.574711</td>
      <td>8.190077</td>
      <td>9.425452</td>
      <td>4.584967</td>
      <td>7.996317</td>
      <td>4.127134</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Milk':



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>10.039983</td>
      <td>11.205013</td>
      <td>10.377047</td>
      <td>6.894670</td>
      <td>9.906981</td>
      <td>6.805723</td>
    </tr>
    <tr>
      <th>98</th>
      <td>6.220590</td>
      <td>4.718499</td>
      <td>6.656727</td>
      <td>6.796824</td>
      <td>4.025352</td>
      <td>4.882802</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>356</th>
      <td>10.029503</td>
      <td>4.897840</td>
      <td>5.384495</td>
      <td>8.057377</td>
      <td>2.197225</td>
      <td>6.306275</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Grocery':



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Frozen':



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>8.431853</td>
      <td>9.663261</td>
      <td>9.723703</td>
      <td>3.496508</td>
      <td>8.847360</td>
      <td>6.070738</td>
    </tr>
    <tr>
      <th>57</th>
      <td>8.597297</td>
      <td>9.203618</td>
      <td>9.257892</td>
      <td>3.637586</td>
      <td>8.932213</td>
      <td>7.156177</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>145</th>
      <td>10.000569</td>
      <td>9.034080</td>
      <td>10.457143</td>
      <td>3.737670</td>
      <td>9.440738</td>
      <td>8.396155</td>
    </tr>
    <tr>
      <th>175</th>
      <td>7.759187</td>
      <td>8.967632</td>
      <td>9.382106</td>
      <td>3.951244</td>
      <td>8.341887</td>
      <td>7.436617</td>
    </tr>
    <tr>
      <th>264</th>
      <td>6.978214</td>
      <td>9.177714</td>
      <td>9.645041</td>
      <td>4.110874</td>
      <td>8.696176</td>
      <td>7.142827</td>
    </tr>
    <tr>
      <th>325</th>
      <td>10.395650</td>
      <td>9.728181</td>
      <td>9.519735</td>
      <td>11.016479</td>
      <td>7.148346</td>
      <td>8.632128</td>
    </tr>
    <tr>
      <th>420</th>
      <td>8.402007</td>
      <td>8.569026</td>
      <td>9.490015</td>
      <td>3.218876</td>
      <td>8.827321</td>
      <td>7.239215</td>
    </tr>
    <tr>
      <th>429</th>
      <td>9.060331</td>
      <td>7.467371</td>
      <td>8.183118</td>
      <td>3.850148</td>
      <td>4.430817</td>
      <td>7.824446</td>
    </tr>
    <tr>
      <th>439</th>
      <td>7.932721</td>
      <td>7.437206</td>
      <td>7.828038</td>
      <td>4.174387</td>
      <td>6.167516</td>
      <td>3.951244</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Detergents_Paper':



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>161</th>
      <td>9.428190</td>
      <td>6.291569</td>
      <td>5.645447</td>
      <td>6.995766</td>
      <td>1.098612</td>
      <td>7.711101</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Delicatessen':



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>109</th>
      <td>7.248504</td>
      <td>9.724899</td>
      <td>10.274568</td>
      <td>6.511745</td>
      <td>6.728629</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>137</th>
      <td>8.034955</td>
      <td>8.997147</td>
      <td>9.021840</td>
      <td>6.493754</td>
      <td>6.580639</td>
      <td>3.583519</td>
    </tr>
    <tr>
      <th>142</th>
      <td>10.519646</td>
      <td>8.875147</td>
      <td>9.018332</td>
      <td>8.004700</td>
      <td>2.995732</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>183</th>
      <td>10.514529</td>
      <td>10.690808</td>
      <td>9.911952</td>
      <td>10.505999</td>
      <td>5.476464</td>
      <td>10.777768</td>
    </tr>
    <tr>
      <th>184</th>
      <td>5.789960</td>
      <td>6.822197</td>
      <td>8.457443</td>
      <td>4.304065</td>
      <td>5.811141</td>
      <td>2.397895</td>
    </tr>
    <tr>
      <th>187</th>
      <td>7.798933</td>
      <td>8.987447</td>
      <td>9.192075</td>
      <td>8.743372</td>
      <td>8.148735</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>203</th>
      <td>6.368187</td>
      <td>6.529419</td>
      <td>7.703459</td>
      <td>6.150603</td>
      <td>6.860664</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>233</th>
      <td>6.871091</td>
      <td>8.513988</td>
      <td>8.106515</td>
      <td>6.842683</td>
      <td>6.013715</td>
      <td>1.945910</td>
    </tr>
    <tr>
      <th>285</th>
      <td>10.602965</td>
      <td>6.461468</td>
      <td>8.188689</td>
      <td>6.948897</td>
      <td>6.077642</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>289</th>
      <td>10.663966</td>
      <td>5.655992</td>
      <td>6.154858</td>
      <td>7.235619</td>
      <td>3.465736</td>
      <td>3.091042</td>
    </tr>
    <tr>
      <th>343</th>
      <td>7.431892</td>
      <td>8.848509</td>
      <td>10.177932</td>
      <td>7.283448</td>
      <td>9.646593</td>
      <td>3.610918</td>
    </tr>
  </tbody>
</table>
</div>



```python
from sklearn.covariance import EllipticEnvelope
clf = EllipticEnvelope(contamination=0.1)
clf.fit(log_data)
y_pred = clf.predict(log_data)
i_outliers = [i for i, x in enumerate(y_pred) if x == -1]

print("Outliers "+ str(len(i_outliers)) + " : " + str(i_outliers))

#[ 38  57  65  65  66  66  75  75  81  86  95  96  98 109 128 128 137 142
# 145 154 154 154 161 171 175 183 184 187 193 203 218 233 264 285 289 304
# 305 325 338 343 353 355 356 357 412 420 429 439]
#[ 65  66  75 128 154]

mhl_outliers = np.argsort(clf.mahalanobis(log_data))[-40:]
print(mhl_outliers)

print (np.sort(imgs/outpliers))

multi_outliers = list(set([i for i in outliers if outliers.count(i) > 1]))
print(np.sort(multi_outliers))
```

    Outliers 44 : [65, 66, 71, 75, 88, 95, 96, 98, 109, 128, 137, 141, 142, 145, 154, 161, 174, 177, 183, 184, 185, 187, 190, 204, 218, 228, 233, 253, 272, 275, 285, 289, 333, 338, 343, 357, 358, 384, 402, 403, 412, 429, 430, 435]
    400    384
    401     98
    402    161
    403    430
    404    403
    405    185
    406    343
    407    435
    408     65
    409    174
    410     88
    411    412
    412    429
    413    275
    414    137
    415    190
    416    285
    417    358
    418     96
    419    272
    420     71
    421    154
    422    177
    423    184
    424    183
    425    228
    426    141
    427    233
    428    357
    429    218
    430     66
    431    204
    432    187
    433    128
    434    402
    435     95
    436    109
    437    142
    438    338
    439     75
    dtype: int64
    [ 38  57  65  65  66  66  75  75  81  86  95  96  98 109 128 128 137 142
     145 154 154 154 161 171 175 183 184 187 193 203 218 233 264 285 289 304
     305 325 338 343 353 355 356 357 412 420 429 439]
    [ 65  66  75 128 154]



```python
# choice of 'outliers', 'multi_outliers', 'i_outliers' and 'mhl_outliers'
#outliers_to_be_removed = outliers
outliers_to_be_removed = outliers

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers_to_be_removed]).reset_index(drop = True)


print(len(log_data))
print(len(imgs/outpliers))
print(len(multi_outliers))
print(len(i_outliers))
print(len(mhl_outliers))
print(len(good_data))
```

    440
    48
    5
    44
    40
    398



```python
# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(good_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```

    /anaconda/envs/boston_housing/lib/python2.7/site-packages/ipykernel_launcher.py:2: FutureWarning: pandas.scatter_matrix is deprecated. Use pandas.plotting.scatter_matrix instead
      



![png](imgs/outpput_27_1.png)



```python
g = sb.PairGrid(np.log(good_data))
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)

```


![png](imgs/outpput_28_0.png)


### Question 4
*Are there any data points considered outliers for more than one feature based on the definition above? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why.* 

**Answer:**

Yes there are, and these should be removed from the dataset.

I've also removed all outliers defined by Tukey's method from the dataset, which amounts to 48 data points. I've also tried multiple scenarios: removing only outliers for more than one feature, removing outliers identified by the Elliptic Envelope, and removing outliers which have the highest Mahalanobis distance. 

However, I found out that the best Silhouette scores for Gaussian Mixture clustering happen when the outliers identified by the Tukey Method were removed.

## Feature Transformation
In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

### Implementation: PCA

Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.

In the code block below, you will need to implement the following:
 - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
 - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.


```python
from sklearn.decomposition import PCA 

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=len(good_data.columns), random_state=0)
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)
```


![png](imgs/outpput_33_0.png)


### Question 5
*How much variance in the data is explained* ***in total*** *by the first and second principal component? What about the first four principal components? Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.*  
**Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

**Answer:**

In total, there's about 72% of the variance explained by the first two principal components, and about 92% explained by the first four principal components. 


* An increase in the first dimension corresponds to an increase in spending in mainly Milk, Grocery and Detergents / Paper, and a decrease in this dimension corresponds to a decrease in spending in those categories.
* An increase in the second dimension corresponds to an increased spending in mainly the Fresh, Frozen and Delicatessen categories. A decrease in this dimension corresponds to a decrease in spending in those categories.
* An increase in the third dimension corresponds to a high decrease in spending in Milk and Detergents / Paper, and an increase in spending in mainly the Frozen and Delicatessen categories. 
* An increase in the third dimension corresponds to a high decrease in spending in Milk and Detergents / Paper, and an increase in spending in mainly the Frozen and Delicatessen categories. A decrease in this dimension reverses the direction of the already-mentioned spending trends for this dimension.
* An increase in the fourth dimension corresponds to a high decrease in spending in Frozen and Detergents / Paper, and an increase in spending in mainly the Fresh and Delicatessen categories. A decrease in this dimension reverses the direction of the already-mentioned spending trends for this dimension.


### Observation
Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.


```python
# Display the log-transformed sample data
display(log_samples)

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.814164</td>
      <td>7.143618</td>
      <td>9.954276</td>
      <td>8.589142</td>
      <td>8.336390</td>
      <td>9.579971</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.687630</td>
      <td>10.740670</td>
      <td>11.437986</td>
      <td>6.933423</td>
      <td>10.617099</td>
      <td>7.987524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.241602</td>
      <td>8.152774</td>
      <td>8.868132</td>
      <td>9.713416</td>
      <td>6.656727</td>
      <td>6.822197</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
      <th>Dimension 3</th>
      <th>Dimension 4</th>
      <th>Dimension 5</th>
      <th>Dimension 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6717</td>
      <td>2.4862</td>
      <td>0.6140</td>
      <td>0.4063</td>
      <td>-1.9952</td>
      <td>1.4342</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.5240</td>
      <td>1.1625</td>
      <td>-0.6076</td>
      <td>0.5663</td>
      <td>0.4874</td>
      <td>0.2678</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.3980</td>
      <td>2.8829</td>
      <td>-1.0725</td>
      <td>-0.9247</td>
      <td>0.3976</td>
      <td>0.3645</td>
    </tr>
  </tbody>
</table>
</div>


### Implementation: Dimensionality Reduction
When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.

In the code block below, you will need to implement the following:
 - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
 - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
 - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.


```python
# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2, random_state=0)
pca.fit(good_data)


# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
```

### Observation
Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.


```python
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6717</td>
      <td>2.4862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.5240</td>
      <td>1.1625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.3980</td>
      <td>2.8829</td>
    </tr>
  </tbody>
</table>
</div>


## Visualizing a Biplot
A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.

Run the code cell below to produce a biplot of the reduced-dimension data.


```python
# Create a biplot
vs.biplot(good_data, reduced_data, pca)

#[[ 2.1349413  -0.28909267]
# [-1.2422204   0.16820922]]
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1149e65d0>




![png](imgs/outpput_43_1.png)


### Observation

Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 

From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

## Clustering

In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

### Question 6
*What are the advantages to using a K-Means clustering algorithm? What are the advantages to using a Gaussian Mixture Model clustering algorithm? Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?*

**Answer:**

K-means clustering algorithm makes a "hard" assignment of data points to a certain cluster. So one data point belongs to one and only one cluster (although this assignment might be revised with every iteration of the algorithm).

On the other hand, a Gaussian Mixture model assigns probabilities to each point, with each probability being the likelihood of belonging to a certain cluser. This is a "soft" assignment.

After seeing the plotted data, I would probably use the Gaussian Mixture model, given that there is no clear separation of clusters, and so "soft" assignments of clusters might make more sense, and since one specific customer can have two "profiles" (or more) at the same time, and so can be targeted by multiple marketing campaigns.

### Implementation: Creating Clusters
Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.

In the code block below, you will need to implement the following:
 - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
 - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
 - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
 - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
 - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
   - Assign the silhouette score to `score` and print the result.


```python
# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

max_clusters = 2

for gaussian in (False, True): 
    for n_clusters in range (2, max_clusters + 1):

        if gaussian == True:
            clusterer = GaussianMixture(n_components=n_clusters, n_init=10, random_state=0)
        else:
            clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)

        clusterer.fit(reduced_data)

        # TODO: Predict the cluster for each data point
        preds = clusterer.predict(reduced_data)

        # TODO: Find the cluster centers

        if gaussian == True:
            centers = clusterer.means_
        else:
            centers = clusterer.cluster_centers_
        #print("Centers: " + str(centers))

        # TODO: Predict the cluster for each transformed sample data point
        sample_preds = clusterer.predict(pca_samples)

        # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
        from sklearn.metrics import silhouette_score
        score = silhouette_score(reduced_data, preds, random_state=0)
        print("Silhouette Score - " + ("K-means", "Gaussian Mixture")[gaussian] + " for " + str(n_clusters) + " clusters: " + str(score))
```

    Silhouette Score - K-means for 2 clusters: 0.447157742293
    Silhouette Score - Gaussian Mixture for 2 clusters: 0.447411995571


### Question 7
*Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?* 

**Answer:**

Simulation Runs: 
* Silhouette Score - K-means for 2 clusters: 0.447157742293
* Silhouette Score - K-meansfor 3 clusters: 0.36398647984
* Silhouette Score - K-means for 4 clusters: 0.331150954285
* Silhouette Score - K-means for 5 clusters: 0.352412352774
* Silhouette Score - K-means for 6 clusters: 0.362761015127
* Silhouette Score - K-means for 7 clusters: 0.354716184747
* Silhouette Score - K-means for 8 clusters: 0.367260764215
* Silhouette Score - K-means for 9 clusters: 0.367312031027
* Silhouette Score - Gaussian Mixture for 2 clusters: 0.447411995571
* Silhouette Score - Gaussian Mixture for 3 clusters: 0.36214071843
* Silhouette Score - Gaussian Mixturefor 4 clusters: 0.306641085838
* Silhouette Score - Gaussian Mixture for 5 clusters: 0.313568743105
* Silhouette Score - Gaussian Mixture for 6 clusters: 0.286150470372
* Silhouette Score - Gaussian Mixture for 7 clusters: 0.3332659059
* Silhouette Score - Gaussian Mixture for 8 clusters: 0.305771875021
* Silhouette Score - Gaussian Mixture for 9 clusters: 0.30935691257


Results explanation:
* For 2 clusters, the score is 0.447 for both k-means and Gaussian Mixture
* For 3 clusters, the score is 0.36 for both k-means and Gaussian Mixture
* For 4 clusters, the score is 0.30 and 0.33 for both k-means and Gaussian Mixture respectively

Both Gaussian Mixture and K-means performances are quite similar, with K-means performing slightly better with the Silhouette score. 

It seems that a number of 2 clusters would be the best clustering, with the highest Silhouette score of 0.447.

### Cluster Visualization
Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 


```python
# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)
```


![png](imgs/outpput_53_0.png)


### Implementation: Data Recovery
Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.

In the code block below, you will need to implement the following:
 - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
 - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.



```python
# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

sb.heatmap((true_centers-data.mean())/data.std(ddof=1), annot=True, cbar=False, square=True)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>5174.0</td>
      <td>7776.0</td>
      <td>11581.0</td>
      <td>1068.0</td>
      <td>4536.0</td>
      <td>1101.0</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>9468.0</td>
      <td>2067.0</td>
      <td>2624.0</td>
      <td>2196.0</td>
      <td>343.0</td>
      <td>799.0</td>
    </tr>
  </tbody>
</table>
</div>





    <matplotlib.axes._subplots.AxesSubplot at 0x114d2bf10>




![png](imgs/outpput_55_2.png)


### Question 8
Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project. *What set of establishments could each of the customer segments represent?*  
**Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`.

**Answer:**

The clusters seem to be positioned horizontally on the Dimension 1 axis, which makes sense since Dimension 1 is highly weighted in Groceries, Milk and Detergents/Paper. A high value in Dimension 1 might suggest a certain type of establishment like a grocery store or a supermarket, and a low value of this Dimension, might suggest otherwise (restaurant or coffee shop). This might not be true for Dimension 2, since some supermarkets will sell Fresh and Frozen goods. And so the clustering will be determined by the values of Dimension 1. 

Segment 0 (high Dimension 1 value) would best be described as a grocery store or supermarket, with a higher than average (above mean) spending on Milk, Grocery, and Detergents/Paper, as demonstrated by the cluster center of this segment in the heatmap above. 

Segment 1 (low Dimension 1 value) would best be described as an establishment such as restaurants and coffee shops. The center of the cluster of customers in this segment, the spending for Fresh, Frozen, and Deli is close to their respective means, but with a relatively lower spending in Milk, Grocery and Detergents_Paper.



### Question 9
*For each sample point, which customer segment from* ***Question 8*** *best represents it? Are the predictions for each sample point consistent with this?*

Run the code block below to find which cluster each sample point is predicted to be.


```python
# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred
    
display(samples)
print(clusterer.predict_proba(pca_samples))
```

    Sample point 0 predicted to be in Cluster 0
    Sample point 1 predicted to be in Cluster 0
    Sample point 2 predicted to be in Cluster 1



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18291</td>
      <td>1266</td>
      <td>21042</td>
      <td>5373</td>
      <td>4173</td>
      <td>14472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16117</td>
      <td>46197</td>
      <td>92780</td>
      <td>1026</td>
      <td>40827</td>
      <td>2944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76237</td>
      <td>3473</td>
      <td>7102</td>
      <td>16538</td>
      <td>778</td>
      <td>918</td>
    </tr>
  </tbody>
</table>
</div>


    [[  6.78845332e-01   3.21154668e-01]
     [  9.99944845e-01   5.51549908e-05]
     [  1.52672169e-02   9.84732783e-01]]


**Answer:**

Given the clustering predictions, it seems straight-forward deciding for the second (grocery store most likely) and third (restaurant) points with consistent predictions (and consistent with the answer to Question 1), but hard to decide for the first sample point. As shown in the graphs, this point is actually on the boundary between the two clusters, and is assigned to Cluster 0. 

As shown, the probability assignment for this point to Cluster 0 is 0.67, and is 0.32 to Cluster 1. So this is close, and "almost" in the boundary between the two. For this data point, the establishment could be of any type, or even a hybrid (a grocery store with a small deli restaurant or coffee shop in it).


## Conclusion

In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

### Question 10
Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. *How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*  
**Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

**Answer:**


Hypothesis:

* The delivery schedule would affect most those customers who order large quantities of perishable products and who don't have the right storage facilities for a less frequent delivery schedule: mainly Fresh products (bread? etc..). This corresponds to Segment 1.

* The customers who order lots of non-perishable (groceries, long-life milk?, detergents / paper) or frozen products will be least affected. So we can use the customer segmentation to start with this class of customers first. This corresponds to Segment 0, which are most probably grocery stores and supermarkets, which means they're specialized in the safe storage and stocking of food and perishables, and would be fine with a changed delivery schedule. 

Given the above hypothesis, how can we validate it using A/B testing? I would think that we could take a random sample from the whole customer population (5%, 10% or 20% of the total population), and test this new delivery schedule on them. Let's call them the "new schedule" group. The remaining majority group are the "control" group. 

After testing the new schedule for a certain number of weeks, we review again the orders made by the "new schedule" group, vs. the "control" group. If we found out statistically different measures of means (using confidence intervals) between the two groups, then we can make a recommendation. For example, if the means of spending in all categories of the "new schedule" group have improved or stayed the same, we could say that the new delivery schedule has positive impact or no impact. Given that this is a cost-cutting measure for the distributor, a "no impact" scenario is still good. This will mean that a reduced delivery schedule will not affect their business.

If however, we found out a negative impact on the means, we could further break down the "new schedule" group into Segment 0 and Segment 1, and see if the means are statistically different between them, and between the "control" group, or the cluster means of the "control" group. It might be the case where there's no impact of the new schedule delivery on Segment 0, but has a negative impact on Segment 1. Therefore, the combination of the two will result in an overall negative impact. 

If we found out for example that for Segment 0, there's positive or no impact, then we prove the hypothesis above. If not, then we need to come up with a new hypothesis and test it. 



### Question 11
Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
*How can the wholesale distributor label the new customers using only their estimated product spending and the* ***customer segment*** *data?*  
**Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

**Answer:**

As suggested in the hint, a supervised learner could be used to train on the original customers, with the target variable being the derived customer segment label (the clustering results). The learner would then be used to predict the customer segment label for these 10 new customers. 

### Visualizing Underlying Distributions

At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.

Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.


```python
# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)
```


![png](imgs/outpput_68_0.png)


### Question 12
*How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*

**Answer:**

This coincides perfectly with the results of our clustering algorithm, and with the derived PCA dimensions. As seen in the above graph, there two clusters are not cleanly separated, and many green and red points are inter-mingling, especially on the boundary between the 2 clusters. Therefore, it would be very hard to say that these customers are purely retailers or purely Hotels/Restaurants/Cafes. 
And again, this is why the Gaussian Mixture clustering algorithm might make more sense, since it will assign "soft" labels and probabilities of belonging to such clusters. Customers with non-insignificant probabilities for both clusters might be listed to be targeted by two marketing campaings, each designed for a certain customer segment. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
