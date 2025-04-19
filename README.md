## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/96758959-e28e-48ae-8349-a5f07df70cb3)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/ce0fb448-3e67-437e-ba67-624391a4ff12)
```
df.describe()
```
![image](https://github.com/user-attachments/assets/ec995c08-5235-447e-9573-e3e10fd0b29b)
```
df.info()
```
![image](https://github.com/user-attachments/assets/8da90fbe-fc1d-4c42-b69d-aa4778d0f580)
```
df
```
![image](https://github.com/user-attachments/assets/738efd7e-6de8-4743-807b-38ebdeffe891)
```
#ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/8f89c99c-4f5f-4883-8266-14eb1ca4f83e)
```
df['bo2']=oe.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/b07bdfbb-2cf2-47d7-a558-9d7a4b557657)
```
#label Encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/d285d7a9-6c83-4936-9514-8b5fc764719a)
```
#One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/8fe1ad7f-b92a-4841-9b55-265be72e8516)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/3437590f-2031-403e-95aa-ef424687e608)
```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df= pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/1aecf1d2-430e-42ba-b95a-27a25b237e83)
```
#binary encoder
be = BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/18e9f115-ef26-4606-b465-9b0d598a1aac)
```
#target encoder
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/bee70402-24eb-4645-a4f8-9257fc1d814f)
```
#Feature Transformation
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/318454f5-a3bd-407c-b7c9-3eed98249aa9)
```
df.info()
```
![image](https://github.com/user-attachments/assets/0e928b84-f33e-456c-93f0-fc36f9dbd442)
```
df.describe()
```
![image](https://github.com/user-attachments/assets/5c02dc55-3624-40d9-bd8c-a55dfa699a3c)
```
df.size
df.skew()
```
![image](https://github.com/user-attachments/assets/abef0919-c432-42d9-b7fa-bc8d8422471c)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/eae73b80-dbdf-493b-aa56-22a69de1ec85)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/bb98585f-c98a-4283-8916-3779e8e825f9)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1281c521-7088-4fcc-a893-e3e5acaef145)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/fa19e42e-1e41-4349-a738-6a148ee97709)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/3132bbae-831f-46d3-886b-8ae22cfb7402)
```
df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/7cf24d96-99f7-4905-be1a-34927d75f35d)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/fff2ab8b-c788-4407-9a12-eaf25e379017)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/697de9e6-4041-4db1-af0b-83d6580d62f2)
```
df
```
![image](https://github.com/user-attachments/assets/022d10d3-86aa-4784-bc4c-8e3027b8a542)
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/768c7107-5298-466f-af6b-06518839b791)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8dec98d8-033e-47d9-bc3d-fef788fa4761)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a6254276-d8b9-456d-ad8d-b8e5f45a09d2)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/0e374b03-473e-49ea-839d-66955d6e7cb9)



# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
