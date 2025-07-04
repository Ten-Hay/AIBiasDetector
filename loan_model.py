#imports
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap
shap.initjs()
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate


#Fetch data
df = pd.read_csv("loan_access_data.csv", encoding="latin-1")
test = pd.read_csv("test.csv", encoding="latin-1")

#Preprocessing data
df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})

X = df.drop(["Loan_Approved", "ID"], axis=1)
y = df["Loan_Approved"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

'''
#Hyperparamter Tunning
rfTuner = RandomForestClassifier(random_state=1)

#Possible Parameters:
param_grid = {
    'n_estimators': [100, 500, 1000],          
    'criterion': ['gini', 'entropy'],     
    'min_samples_split': [2, 5, 10],        
    'max_depth': [None, 10, 20, 30],         
    'min_samples_leaf': [1, 2, 4]            
}

grid_search = GridSearchCV(
    estimator=rfTuner,
    param_grid=param_grid,
    cv=5,              
    scoring='accuracy', 
    n_jobs=-1,          
    verbose=2         
)


grid_search.fit(X_train, y_train)

print("Hyperparameters:", grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)
'''

#Labeling Sensitive Features
sensitive_features = ["Gender", "Race", "Employment_Type", "Education_Level", 
                      'Citizenship_Status', 'Language_Proficiency', 
                      'Zip_Code_Group', 'Criminal_Record', "Disability_Status"]

#Creating raw dat audit charts
for feature in sensitive_features:
    plt.figure(figsize=(6, 4))
    
    approval_rate_df = df.groupby(feature)['Loan_Approved'].mean().reset_index()
    approval_rate_df.columns = [feature, 'ApprovalRate']
    approval_rate_df = approval_rate_df.sort_values(by='ApprovalRate', ascending=False)

    sns.barplot(
        data=approval_rate_df,
        x=feature,
        y='ApprovalRate',
        hue=feature,
        palette="viridis",
        dodge=False,
        legend=False 
    )
    
    plt.ylabel("Approval Rate")
    plt.xlabel(feature)
    plt.title(f"Loan Approval Rate by {feature}")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.tight_layout()
    filename = f"charts/raw-data-audit/{feature}_approval_rate.png"
    plt.savefig(filename)
    plt.close()



#Creating Model With Updated Hyperparameters
rf = RandomForestClassifier(
    n_estimators=500,
    criterion="entropy",
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=10
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

#Printing scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


#Model Bias Audit Using Shap
explainer = shap.Explainer(rf)
shap_values = explainer(X_test)
np.shape(shap_values.values)


print("SHAP values shape:", shap_values.values.shape)

#Shap Beeswarm Graph
shap.summary_plot(shap_values.values[:, :, 1], X_test, show=False)
plt.tight_layout()
plt.savefig( 'charts/model-data-audit/shap/summary.png')
plt.clf()

#SHAP individual waterfalls (selected based upon known sensitive features)
shap.plots.waterfall(shap_values[:, :, 1][0], show=False)
plt.tight_layout()
plt.savefig( 'charts/model-data-audit/shap/waterfall1.png')
plt.clf()

shap.plots.waterfall(shap_values[:, :, 1][9], show=False)
plt.tight_layout()
plt.savefig( 'charts/model-data-audit/shap/waterfall2.png')
plt.clf()

shap.plots.waterfall(shap_values[:, :, 1][17], show=False)
plt.tight_layout()
plt.savefig( 'charts/model-data-audit/shap/waterfall3.png')
plt.clf()

plt.close()



#Label metrics to be graphed
metrics = {
    "selection_rate": selection_rate,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
}

#Colors for difference
default_color = "lightblue"
highlight_color = "red"

#Iterate through features
for feature in sensitive_features:
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=df.loc[X_test.index, feature]
    )
    
    results_df = metric_frame.by_group.reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    
    #Create each metric barplot for the feature
    for ax, metric_name in zip(axes, metrics.keys()):
        values = results_df[metric_name]
        groups = results_df[feature]
        
        #Estalbish value for different color
        threshold = values.mean() + (values.std()/2)
        color_map = {
        group: highlight_color if value > threshold else default_color
        for group, value in zip(groups, values)
        }
        
        #Barplot setup
        sns.barplot(
            data=results_df,
            x=feature,
            y=metric_name,
            order=results_df[feature],
            hue=feature,
            palette=color_map,
            ax=ax
        )
        ax.set_title(f"{metric_name.replace('_', ' ').title()} by {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        
        ax.axhline(threshold, ls='--', color='gray', label=f"Threshold ({threshold:.2f})")
        ax.legend()
    
    #Save feature
    plt.tight_layout()
    filename = f"charts/model-data-audit/metrics/{feature}_metrics.png"
    plt.savefig(filename)
    plt.close()
    

#Creating sumbission.csv
test_features = test.drop(["ID"], axis=1) 

#Preproces test data
test_dummies = pd.get_dummies(test_features)
test_dummies = test_dummies.reindex(columns=X.columns, fill_value=0)

test_preds = rf.predict(test_dummies)

submission = pd.DataFrame({
    "ID": test["ID"],  
    "LoanApproved": test_preds
})

submission.to_csv("submissions.csv", index=False)

