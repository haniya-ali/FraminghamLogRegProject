# FraminghamLogRegProject

https://mendedhearts.org/wp-content/uploads/2017/03/MEN-web-CHD-Facts-and-Statistics.jpg

**Introduction:**

![image](https://github.com/haniya-ali/FraminghamLogRegProject/assets/79181650/c0764c93-fab1-4103-9663-c567b4b19507)

**Congenital heart defects (CHDs)**, also known as congenital heart abnormalities, are medical conditions that can impact the structure and function of an individual's heart. These defects occur when a baby's heart does not develop normally during pregnancy. They are considered the most common type of birth defect, as stated by the National Library of Medicine.

According to estimates from the Centers for Disease Control and Prevention (CDC), CHDs affect nearly 1% of births in the United States, which accounts for approximately **40,000 births per year**. It's important to note that the prevalence of CHDs can vary depending on the specific type of defect.

Over time, there has been an observed increase in the number of babies born with certain types of heart defects, particularly milder forms, when compared to the overall number of births. On the other hand, the prevalence of other types of congenital heart defects has remained stable.

It is crucial to raise awareness about CHDs and ensure early detection, accurate diagnosis, and appropriate treatment options to improve the outcomes for individuals with congenital heart defects.

**Abstract:**
During the late 1940s, the United States government initiated a significant endeavor to explore cardiovascular disease. To ensure the generation of high-quality data, they made the strategic decision to monitor a large group of initially healthy individuals over an extended period. The chosen location for this study was Framingham, Massachusetts, a suburb of Boston, where the project commenced in 1948. A total of **5,209 participants**, aged between 30 and 62, were enrolled in the study.

At regular intervals of two years, participants underwent both a medical examination and responded to a questionnaire. In addition to gathering medical test data, the study also captured information related to the participants' physical and behavioral characteristics. As time progressed, the study expanded its scope, encompassing multiple generations and incorporating numerous additional factors, including genetic information. This extensive dataset is now widely recognized as the Framingham Heart Study, an illustrious and influential resource within cardiovascular research.

In this project, I will utilize the data from the **Framingham Heart Study** to predict coronary heart disease (CHD) and offer recommendations to enhance heart disease prevention. The dataset consists of 3,658 observations, each representing data from a specific study participant. It comprises 16 variables, and the target variable I aim to predict is **TenYearCHD**, which indicates whether a patient will experience coronary heart disease within ten years of their initial examination. Through our modeling endeavors, I anticipate identifying risk factors, which are the variables that heighten the risk of developing CHD.

The primary model that will be employed for this project will be a **logistic regression model** because it is well-suited for predicting binary outcomes like the occurrence of coronary heart disease within ten years.

**Conclusions:**
In order to offer recommendations to enhance heart disease prevention, this project used the Framingham Heart Study data to develop a logistic regression model that can predict how likeley someone is to develop coronary heart disease within ten years. The model accuracy was ~85% on the test set. 

Our analysis also revealed that the most important risk factors for predicting CHD are:

i. age

ii. sysBP

iii. heartRate

iv. glucose

v. gender (male)

**Further considerations:**
Despite the dataset containing 16 independent variable, the dataset does not contain any information on the racial distribution of the sample of indivdiuals that it was collected from. Some medical conditions like blood pressure, diabetes, etc are more prevalent in certain populations. As such it is essential for a predictive model that uses independent variables to predict the risk of developing CHD to include some metric of determining the racial, ethnic, or geographic identity. The consequence of not having this information is that we may have trained the model on indivdiuals in the US where serving sizes are larger. If we use this model to predict CHD on indivduals in another country, we may not get a correct representation of the underlying trends.

The analysis could be imporved by identifying the exact population from which the samples were derived. This consistency would allow us to ensure that we are training and testing the model on the appropriate individuals.

