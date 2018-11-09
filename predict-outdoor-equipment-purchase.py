
# coding: utf-8

# In[1]:

sc


# In[2]:

spark


# ## Learning goals
# 
# The learning goals of this notebook are:
# 
# -  Load a CSV file into an Apache® Spark DataFrame.
# -  Explore data.
# -  Prepare data for training and evaluation.
# -  Create an Apache® Spark machine learning pipeline.
# -  Train and evaluate a model.
# -  Persist a pipeline and model in Watson Machine Learning repository.
# -  Deploy a model for online scoring using Wastson Machine Learning API.
# -  Score sample scoring data using the Watson Machine Learning API.
# -  Explore and visualize prediction result using the plotly package.
# 
# 
# ## Contents
# 
# This notebook contains the following parts:
# 
# 1.	[Setup](#setup)
# 2.	[Load and explore data](#load)
# 3.	[Create spark ml model](#model)
# 4.	[Persist model](#persistence)
# 5.	[Predict locally and visualize](#visualization)
# 6.	[Deploy and score in a Cloud](#scoring)
# 7.	[Summary and next steps](#summary)

# <a id="load"></a>
# ## 1. Load and explore data

# In this section you will load the data as an Apache® Spark DataFrame and perform a basic exploration.
# 
# Load the data to the Spark DataFrame by using *wget* to upload the data to gpfs and then *read* method. 

# In[3]:

df_data = spark.read  .format('csv')  .option('header', 'true')  .option('inferSchema', 'true')  .load('s3://bigdatateaching/misc/gosales_tx_naivebayes.csv')


# Explore the loaded data by using the following Apache® Spark DataFrame methods:
# -  print schema
# -  print top ten records
# -  count all records

# In[ ]:

df_data.printSchema()


# As you can see, the data contains five fields. PRODUCT_LINE field is the one we would like to predict (label).

# In[ ]:

df_data.show()


# In[ ]:

print "Number of records: " + str(df_data.count())


# As you can see, the data set contains 60252 records.

# <a id="model"></a>
# ## 2. Create an Apache® Spark machine learning model
# 
# In this section you will learn how to prepare data, create an Apache® Spark machine learning pipeline, and train a model.

# ### 2.1: Prepare data
# 
# In this subsection you will split your data into: train, test and predict datasets.

# In[ ]:

splitted_data = df_data.randomSplit([0.8, 0.18, 0.02], 24)
train_data = splitted_data[0]
test_data = splitted_data[1]
predict_data = splitted_data[2]

print "Number of training records: " + str(train_data.count())
print "Number of testing records : " + str(test_data.count())
print "Number of prediction records : " + str(predict_data.count())


# As you can see our data has been successfully split into three datasets: 
# 
# -  The train data set, which is the largest group, is used for training.
# -  The test data set will be used for model evaluation and is used to test the assumptions of the model.
# -  The predict data set will be used for prediction.

# ### 2.2: Create pipeline and train a model

# In this section you will create an Apache® Spark machine learning pipeline and then train the model.

# In the first step you need to import the Apache® Spark machine learning packages that will be needed in the subsequent steps.

# In[ ]:

from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, Model


# In the following step, convert all the string fields to numeric ones by using the StringIndexer transformer.

# In[ ]:

stringIndexer_label = StringIndexer(inputCol="PRODUCT_LINE", outputCol="label").fit(df_data)
stringIndexer_prof = StringIndexer(inputCol="PROFESSION", outputCol="PROFESSION_IX")
stringIndexer_gend = StringIndexer(inputCol="GENDER", outputCol="GENDER_IX")
stringIndexer_mar = StringIndexer(inputCol="MARITAL_STATUS", outputCol="MARITAL_STATUS_IX")


# In[ ]:

stringIndexer_label.labels


# In the following step, create a feature vector by combining all features together.

# In[ ]:

vectorAssembler_features = VectorAssembler(inputCols=["GENDER_IX", "AGE", "MARITAL_STATUS_IX", "PROFESSION_IX"], outputCol="features")


# Next, define estimators you want to use for classification. Random Forest is used in the following example.

# In[ ]:

rf = RandomForestClassifier(labelCol="label", featuresCol="features")


# Finally, indexed labels back to original labels.

# In[ ]:

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=stringIndexer_label.labels)


# Let's build the pipeline now. A pipeline consists of transformers and an estimator.

# In[ ]:

pipeline_rf = Pipeline(stages=[stringIndexer_label, stringIndexer_prof, stringIndexer_gend, stringIndexer_mar, vectorAssembler_features, rf, labelConverter])


# Now, you can train your Random Forest model by using the previously defined **pipeline** and **train data**.

# In[ ]:

train_data.printSchema()


# In[ ]:

model_rf = pipeline_rf.fit(train_data)


# You can check your **model accuracy** now. To evaluate the model, use **test data**.

# In[ ]:

predictions = model_rf.transform(test_data)
evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))


# You can tune your model now to achieve better accuracy. For simplicity of this example tuning section is omitted.

# ### Authors
# 
# **Lukasz Cmielowski**, PhD, is a Automation Architect and Data Scientist in IBM with a track record of developing enterprise-level applications that substantially increases clients' ability to turn data into actionable knowledge.

# Copyright © 2017 IBM. This notebook and its source code are released under the terms of the MIT License.

# Adapted from [this notebook from IBM Watson.](https://dataplatform.ibm.com/analytics/notebooks/89492fd6-a641-4819-9176-3d9381561df9/view?access_token=d80bef1a172d1d83d3721b101886337158457281774186f181a2e6a5b57f5ec7#)
