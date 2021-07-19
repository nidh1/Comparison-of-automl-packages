#!/usr/bin/env python
# coding: utf-8

# In[15]:


import mlflow
import mlflow.sklearn


# <b><h1>USING PYCARET

# PyCaret is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within minutes in your choice of notebook environment.

# In[16]:


#mlflow.set_experiment('Comparison of automl packages')
#mlflow.start_run(run_name = "Pycaret")

mlflow.start_run()
mlflow.end_run()
# In[1]:
mlflow.start_run(run_name = "Pycaret")

#opening zip file which contains contains the csv file
import shutil
import os
fil = os.getcwd()
shutil.unpack_archive('fl_cell_big.zip',fil)


# In[2]:


import pandas as pd
df = pd.read_csv('fl_cell_big.csv')
print(df)


# In[3]:


df_mean = df.groupby('date').mean()
df_mean = df_mean.add_suffix('_mean')
df_mean = df_mean.reset_index()

df_std = df.groupby('date').std()
df_std = df_std.add_suffix('_std')
df_std = df_std.reset_index()

df_median = df.groupby('date').median()
df_median = df_median.add_suffix('_median')
df_median = df_median.reset_index()

df_var = df.groupby('date').var()
df_var = df_var.add_suffix('_var')
df_var = df_var.reset_index()

df_min = df.groupby('date').min()
#df_min = df_min.add_suffix('_min')
#df_min = df_min.reset_index()

df_max = df.groupby('date').max()
#df_max = df_max.add_suffix('_max')
#df_max = df_max.reset_index()

df_skew = df.groupby('date').skew()
df_skew = df_skew.add_suffix('_skew')
df_skew = df_skew.reset_index()

df_sem = df.groupby('date').sem()
df_sem = df_sem.add_suffix('_sem')
df_sem = df_sem.reset_index()


df_range = df_max.sub(df_min)
df_min = df_min.add_suffix('_min')
df_max = df_max.add_suffix('_max')
df_max = df_max.reset_index()
df_min = df_min.reset_index()
df_range = df_range.add_suffix('_range')
df_range = df_range.reset_index()


# In[4]:


df_summ=df_mean.merge(df_std,on='date').merge(df_median,on='date').merge(df_var,on='date').merge(df_min,on='date').merge(df_max,on='date').merge(df_skew,on='date').merge(df_sem,on='date').merge(df_range,on='date')


# In[5]:


df_b=df_summ.drop(['% Silica Concentrate_std','% Silica Concentrate_var','% Silica Concentrate_max','% Silica Concentrate_min','% Silica Concentrate_skew','% Silica Concentrate_range','% Silica Concentrate_sem'],axis=1)


# In[6]:


a=df_b['% Silica Concentrate_mean']
b=df_b['date']


# In[7]:


print(df_b)


# In[8]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
sc = StandardScaler()
df_b=df_b.set_index('date')
sc.fit(df_b.drop(['% Silica Concentrate_mean'],axis=1))
std = sc.transform(df_b.drop(['% Silica Concentrate_mean'],axis=1))
pca = PCA(n_components=10)
std=pca.fit_transform(std)
df_c=pd.DataFrame(std)
df_c['% Silica Concentrate_mean']=a
df_c['date']=b
df_c['lag-value']=df_c['% Silica Concentrate_mean'].shift(1)
print(df_c)


# In[9]:


df_c = df_c.dropna()


# In[77]:


import numpy as np
from sklearn.model_selection import train_test_split
train,test = train_test_split(df_c, test_size=0.33, random_state=42)
train.to_csv('train.csv')
test.to_csv('test.csv')


# 1)First install pycaret then import the library of pycaret required for regression analysis.

# In[11]:


#!pip install pycaret
from pycaret.regression import *


# 2)Then use setup to initialize the features. I gave feature_interaction, feature_ratio and polynomial_features as true over here.

# In[12]:


reg1 = setup(data = train.drop('date',axis=1), target = '% Silica Concentrate_mean',polynomial_features = True,feature_interaction = True, feature_ratio = True)


# 3)Then check which model is giving the best performance.

# In[13]:


best = compare_models()


# 4)Create the model.

# In[14]:


lgbm=create_model('lightgbm')


# In[35]:


lab = pd.DataFrame(predict_model(lgbm))['Label']
correct = pd.DataFrame(predict_model(lgbm))['% Silica Concentrate_mean']
corr_matrix = np.corrcoef(correct, lab)
corr=corr_matrix[0,1]
r_sq_train=corr**2
print(r_sq_train)


# In[17]:


mlflow.sklearn.log_model(lgbm,"lgbm_using_pycaret")


# In[36]:


predictions = predict_model(lgbm, data=test.drop(['date'],axis=1))


# In[37]:


predictions['date']=test['date']
predictions=predictions.set_index('date')
predictions=predictions.sort_index()


# In[46]:


import plotly.graph_objects as go
plot_col=["% Silica Concentrate_mean","Label"]
pd.options.plotting.backend = "plotly"
fig = predictions[plot_col].dropna().sort_index().plot()
fig.write_html("lgbm.html")


# 5)The r^2 value for the test data.

# In[39]:


corr_matrix = np.corrcoef(predictions["% Silica Concentrate_mean"], predictions["Label"])
corr=corr_matrix[0,1]
r_sq_test=corr**2
print(r_sq_test)


# In[41]:


mlflow.log_metrics({'rsq_train':r_sq_train, 'rsq_test':r_sq_test})


# In[42]:


import matplotlib.pyplot as plt
fig = plot_model(lgbm,plot="feature")
plt.savefig("feature.jpg",show=True)


# In[43]:


#get_ipython().system('pip install shap')


# In[44]:


import matplotlib.pyplot as plt
fig = interpret_model(lgbm)


# In[48]:


mlflow.log_artifact("lgbm.html")
mlflow.log_artifact("feature.jpg")


# In[49]:


mlflow.end_run()


# In[101]:


import os
model_paths = os.getcwd()
save_model(lgbm, model_paths+'Final_model')


# In[23]:


#!pip install bentoml


# 7)Deploy the model using bentoml, the steps to which are mentioned in this document 

# In[29]:


#from flotation_cell import flotation_cell

# 2) `pack` it with required artifacts
#bento_service = flotation_cell()
#model =  load_model(model_paths+'Final_model')

##saved_path = bento_service.save()


# In[32]:


#!bentoml serve flotation_cell:latest --run-with-ngrok


# <b><h1>USING TPOT

# The Tree-Based Pipeline Optimization Tool (TPOT) was one of the very first AutoML methods and open-source software packages developed for the data science community. TPOT was developed by Dr. Randal Olson while a postdoctoral student with Dr. Jason H. Moore at the Computational Genetics Laboratory of the University of Pennsylvania and is still being extended and supported by this team.
# 
# The goal of TPOT is to automate the building of ML pipelines by combining a flexible expression tree representation of pipelines with stochastic search algorithms such as genetic programming. TPOT makes use of the Python-based scikit-learn library as its ML menu.
# 

# 1)First we need to install tpot, which require an internet connection

# In[35]:


#!pip install tpot


# In[57]:


mlflow.start_run(run_name = "tpot")


# 2)Then do the following steps, it doesn’t require any internet connection, please note that this process takes a lot of time.

# In[36]:


from tpot import TPOTRegressor
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=2, random_state=1)
tpot = TPOTRegressor(generations=0,population_size=1,scoring='r2',cv=cv,verbosity=2,random_state=1,n_jobs=-1)
tpot.fit(train.drop(["% Silica Concentrate_mean","date"],axis=1),train['% Silica Concentrate_mean'])


# 3)Then export the best pipeline code into a python file, no internet connection required.

# In[ ]:


tpot.export('tpot_flotation_best_model.py')


# In[50]:



# In[56]:


from sklearn.model_selection import cross_val_score
rsq_train = cross_val_score(exported_pipeline,train.drop(["% Silica Concentrate_mean","date"],axis=1),train['% Silica Concentrate_mean'],scoring='r2').mean()


# 4)r^2 value can be found out using the following command

# In[60]:


model.score(test.drop(["% Silica Concentrate_mean","date"],axis=1),test['% Silica Concentrate_mean'])

sklearn_pipeline = tpot.fitted_pipeline_

from sklearn.model_selection import cross_val_score
rsq_train = tpot._optimized_pipeline_score


# In[61]:


mlflow.log_metrics({'rsq_train':rsq_train, 'rsq_test':model.score(test.drop(["% Silica Concentrate_mean","date"],axis=1),test['% Silica Concentrate_mean'])})


# In[65]:


predict = sklearn_pipeline.predict(test.drop(["% Silica Concentrate_mean","date"],axis=1))


# In[69]:


import plotly.graph_objects as go
predictions=test
predictions['predict']=predict
predictions=predictions.set_index('date')
predictions=predictions.sort_index()
plot_col=["% Silica Concentrate_mean","predict"]
pd.options.plotting.backend = "plotly"
fig = predictions[plot_col].dropna().sort_index().plot()
fig.write_html("gbr.html")


# In[70]:


mlflow.sklearn.log_model(sklearn_pipeline,"gbr_using_tpot")


# In[71]:


mlflow.log_artifact("gbr.html")
mlflow.log_artifact("tpot_flotation_best_model.py")


# In[72]:


mlflow.end_run()


# 5)Now we can use bentoml to save the model as a REST API

# In[ ]:


##3bento_service = flotation_cell()
#model =  load_model(model_paths+'Final_model')

#bento_service.pack('model', model)

# 3) save your BentoSerivce to file archive
#saved_path = bento_service.save()


# In[ ]:


#!bentoml serve flotation_cell:latest --run-with-ngrok


# <h1><b>USING H2O

# H2O is a Java-based software for data modeling and general computing. 

# In[73]:


mlflow.start_run(run_name="h2o")


# 1)Install h2o, this requires an internet connection

# In[74]:


#conda install -c conda-forge h2o-py openjdk -y


# 2)Import h2o and connect it to a local server, no internet required.

# In[75]:


import h2o
h2o.init()


# 3)When we click on that link we will go to the h2o flow page, no internet required.

# In[78]:


train.drop("date",axis=1).to_csv("train.csv",index=False)
test.drop("date",axis=1).to_csv("test.csv",index=False)


# 4)Now import the train and test data into h2o cluster, by first converting them into csv files, no internet required.

# In[79]:


train1 = h2o.import_file("C:/Users/Hp/train.csv",header=1)
test1 = h2o.import_file("C:/Users/Hp/test.csv",header=1)
print(train1)


# In[80]:


x = train1.columns
y = "% Silica Concentrate_mean"
x.remove(y)


# 5)Then use automl to find out the best model, no internet required.

# In[81]:


from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=x, y=y, training_frame=train1)


# In[102]:


print(aml.leader)


# In[83]:


import mlflow.h2o
mlflow.h2o.log_model(aml.leader,'stackendsemble_using_h2o')


# 6)Then predict the values for test data, no internet required.

# In[84]:


predict = aml.leader.predict(test1.drop("% Silica Concentrate_mean"))


# In[85]:


predict =predict.as_data_frame()


# In[86]:


result = predict["predict"].tolist()


# In[87]:


test['predict'] = result


# In[88]:


corr_matrix = np.corrcoef(test["% Silica Concentrate_mean"], test["predict"])
corr=corr_matrix[0,1]
r_sq=corr**2
print(r_sq)


# In[95]:


import plotly.graph_objects as go
predictions=test
predictions['predict']=predict
predictions=predictions.set_index('date')
predictions=predictions.sort_index()
plot_col=["% Silica Concentrate_mean","predict"]
pd.options.plotting.backend = "plotly"
fig = predictions[plot_col].dropna().sort_index().plot()
fig.write_html("se.html")


# In[96]:


rsq_train = aml.leader.r2()


# In[97]:


mlflow.log_metrics({'rsq_train':rsq_train, 'rsq_test':r_sq})


# In[98]:





# 6)An example of how to get the visualisation of a model using h2o

# In[99]:


import matplotlib.pyplot as plt
pd_plot = aml.leader.ice_plot(test1,"lag-value")
plt.savefig("individual_conditional_expectation_plot.jpeg")

mlflow.log_artifact("individual_conditional_expectation_plot.jpeg")

# In[89]:


import matplotlib.pyplot as plt
pd_plot = aml.leader.residual_analysis_plot(test1)
plt.savefig("residual.jpeg")
mlflow.log_artifact('residual.jpeg')

import matplotlib.pyplot as plt
pd_plot = aml.varimp_heatmap()
plt.savefig("heat_map.jpeg")
mlflow.log_artifact('heat_map.jpeg')

import matplotlib.pyplot as plt
from h2o.estimators import H2OGradientBoostingEstimator
model = H2OGradientBoostingEstimator()
model.train(x=x, y=y, training_frame=train1)
pd_plot = model.shap_summary_plot(test1)
plt.savefig("feature_interaction.jpeg")
mlflow.log_artifact('feature_interaction.jpeg')

import matplotlib.pyplot as plt
pd_plot = aml.model_correlation_heatmap(test1)
plt.savefig("model_correlation.jpeg")
mlflow.log_artifact('model_correlation.jpeg')




# 7)We can get the whole explanation of how the different models work on our data using the below command

# In[106]:


exm = aml.explain(test1)


# In[90]:


mlflow.end_run()


# 9)Lastly we deploy the model using bentoml(make sure that bentoml is installed)

# In[96]:


#get_ipython().run_cell_magic('writefile', 'flotation_cell.py', 'import pandas as pd\nimport bentoml\nfrom bentoml.frameworks.sklearn import SklearnModelArtifact\nfrom bentoml.frameworks.h2o import H2oModelArtifact\nfrom bentoml.service.artifacts.common import PickleArtifact\nfrom bentoml.handlers import DataframeHandler\nfrom bentoml.adapters import DataframeInput\n\n@bentoml.artifacts([H2oModelArtifact(\'model\')])\n@bentoml.env(pip_packages=["scikit-learn", "pandas","h2o"],\n            conda_channels=[\'h2oai\'],\n            conda_dependencies=[\'h2o\'])\nclass flotation_cell(bentoml.BentoService):\n\n    @bentoml.api(input=DataframeInput(), batch=True)\n    def predict(self, df):\n        """\n        predict expects pandas.Series as input\n        """        \n#         series = df.iloc[0,:\n        hf = h2o.H2OFrame(df)\n        return self.artifacts.model.predict(hf).as_data_frame()')


# In[ ]:


# 1) import the custom BentoService defined above
#from flotation_cell import flotation_cell

# 2) `pack` it with required artifacts
#bento_service = flotation_cell()
#bento_service.pack('model', aml.leader)

# 3) save your BentoSerivce to file archive
#saved_path = bento_service.save()


# In[ ]:


#!bentoml serve flotation_cell:latest --run-with-ngrok


# We can see the comparison of all the packages using the following command since we had logged in all the details using mlflow, since the link is not appearing here, we will have to type the command 'mlflow ui' in anaconda prompt to get the link of mlflow.

# In[113]:


#get_ipython().system('jupyter nbconvert -- to python Comparison_of_automl.ipynb')


# In[ ]:




