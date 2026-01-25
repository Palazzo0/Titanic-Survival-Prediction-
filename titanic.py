import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#CLEANING UP THE DATA
data = pd.read_csv("train.csv")

#CREATING A PIPELINE/SPLITING TRAIN AND TEST SETS
X = data.drop(columns = ["Survived", "Ticket", "Name", "PassengerId", "Cabin"])
Y = data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state =42)

numerical_x = X.drop(columns =["Embarked", "Sex"])
categorical_x = X[["Embarked", "Sex"]]

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers= [("num", num_pipeline, list(numerical_x)), ("cat", cat_pipeline,list(categorical_x))])
model_pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])


#BUILD MODEL
model = model_pipeline.fit(x_train, y_train)


#TEST MODEL ACCURACY 
#1- Using the jaccard score
model_accuracy = model.score(x_test, y_test)
#print(f"Accuracy: {model_accuracy}")

#2- Using confusion meatrix
#USE MODEL TO PREDICT Ŷ
y_predict = model.predict(x_test)
#print(confusion_matrix(y_test, y_predict))
#print(classification_report(y_test, y_predict))

#EXTRACT FEATURE IMPORTANCE.
#first of all extract features from Pipeline
preprocessor = model.named_steps["preprocessing"]

cat_x_features = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out()
)

feature_names = list(numerical_x) + list(cat_x_features)

#get feature importance
coef = model.named_steps["classifier"].coef_[0]
#create a table for it
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coef
}).sort_values(by="Coefficient", ascending=False)

#print(feature_importance)

#Visualize the table
import matplotlib.pyplot as plt

feature_importance.set_index("Feature").plot(
    kind="barh",
    figsize=(10, 8),
    legend=False
)

plt.tight_layout()
plt.savefig("feature_importance.png")

#USE MODEL TO MAKE PREDICTION OF THE TEST.CSV
#1- read,clean, and analyse
data_pred = pd.read_csv("test.csv")
data_pred.drop(columns = ["Cabin"], inplace=True)
PassengerId = data_pred["PassengerId"]
#Predict
survival = model.predict(data_pred)
survival_df = pd.DataFrame({"PassengerId": PassengerId,
  "Survived": survival
})
survival_df.to_csv("Palazzo's Titanic_survival prediction.csv", index = False)




print(len(survival))