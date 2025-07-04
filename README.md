# Problem Overview

The task that was given was to create a model that can predict loan approvals based upon a large dataset that was given.
This is an increasingly important real world problem since AI has become such a prominent feature in our rapidly changing world. We have to ensure that the data and models we use are fair across demographics because if not this would lead to biased advantages towards specific groups.


## Model Summary

The model that I used was Random Forest Classifier. I felt this was the most suitable choice as for starters this was a classification problem so using a classification machine learning method was important. Due to the limited data set, Random Forest’s method of combining predictions from multiple trees helped reduce overfitting and improved the accuracy compared to when I tried to use TensorFlow which consistently overfitted the data. Along with this the model required minimal preprocessing since I did not have to normalize or scale the data.

## Bias Considerations

The dataset that was provided was a table of loan approval data which contained many applicants and their data and whether or not they had been approved or denied. This data had a lot of sensitive attributes that are commonly biased against including race, gender, and citizenship status just to name a few.

# Link to Video Submission
https://youtu.be/fmtwx8SDwzg

# Setup

### Clone the repository
```
git clone https://github.com/Ten-Hay/AIBiasDetector
cd AIBiasDetector
```

### Create a virtual environment (recommended but optional)
```
python -m venv venv
source venv/bin/activate
```
### Install dependencies
```
pip install pandas scikit-learn matplotlib seaborn shap fairlearn
```

### Run code
```
python load_model
```
