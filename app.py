# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Disable the warning about the use of Matplotlib's global figure object
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
dataset_path = 'Dataset.csv\\New_Dataset1.csv'
dataset = pd.read_csv(dataset_path)

# Label encode categorical columns
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['Screen Time'] = le.fit_transform(dataset['Screen Time'])
dataset['Do you have siblings?'] = le.fit_transform(dataset['Do you have siblings?'])
dataset['Type of Usage'] = le.fit_transform(dataset['Type of Usage'])
dataset['Do you do any extra curricular activities?'] = le.fit_transform(dataset['Do you do any extra curricular activities?'])
dataset['Do you have a dedicated time to spend with family?'] = le.fit_transform(dataset['Do you have a dedicated time to spend with family?'])

# Train-test split
X = dataset.drop(['sum', 'Presence or no presence of mental issues'], axis=1)  # Assuming 'sum' is the target variable
y = dataset['Presence or no presence of mental issues']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app header with custom styling
st.title("🌈 Mental Health Analysis and ML Model Evaluation 🧠")
st.markdown(
    "Explore visualizations and evaluate machine learning models on a dataset related to mental health."
)

# Display the dataset with a styled title
st.subheader("Raw Data Overview")
st.dataframe(dataset)

# Sidebar for user interaction
st.sidebar.header("Choose Visualization Options")

# Dropdown for selecting a column for bar chart
selected_column = st.sidebar.selectbox("Select a Column for Bar Chart", dataset.columns)

# Bar chart with styled title
st.subheader(f"Bar Chart for {selected_column}")
bar_chart_data = dataset[selected_column].value_counts()
st.bar_chart(bar_chart_data)

# Histogram with styled title
# Histogram with styled title
st.subheader("Histogram")
hist_fig, hist_ax = plt.subplots(figsize=(8, 6))
hist_ax.hist(dataset['Age'], bins=len(dataset['Age']), edgecolor='k', align='left')
hist_ax.set_title('Age Distribution of Survey Respondents')
hist_ax.set_xlabel('Age Range')
hist_ax.set_ylabel('Number of Respondents')
hist_ax.set_xticks(hist_ax.get_xticks())
hist_ax.set_xticklabels(hist_ax.get_xticks(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
st.pyplot(hist_fig)


# Bar chart using Seaborn with styled title
st.subheader("Bar Chart using Seaborn")
bar_seaborn_fig, bar_seaborn_ax = plt.subplots()
sns.barplot(x='Screen Time', y='Feeling down, depressed or hopeless', data=dataset, ax=bar_seaborn_ax)
bar_seaborn_ax.set_title('Count of Feeling down, depressed or hopeless by Screen Time')
bar_seaborn_ax.set_xlabel('Screen Time')
bar_seaborn_ax.set_ylabel('Count')
st.pyplot(bar_seaborn_fig)

# Explore the data with styled title
# Visualize the relationship between screen time and mental health indicators
# Correlation Heatmap with styled title
st.subheader("Correlation Heatmap")
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
st.pyplot(plt.gcf())

# Scatter plot for the relationship between screen time and mental health indicators with styled title
st.subheader("Relationship between Screen Time and Mental Fatigue")
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Screen Time', y='Feeling tired or having little energy', data=dataset)
plt.xlabel('Screen Time')
plt.ylabel('Feeling tired or having little energy')
st.pyplot()

# Boxplot for the relationship between 'Screen Time' and 'Presence or no presence of mental issues' with styled title
st.subheader("Relationship between Screen Time and Mental Issues")
plt.figure(figsize=(12, 6))
sns.boxplot(x='Presence or no presence of mental issues', y='Screen Time', data=dataset)
plt.xlabel('Presence or no presence of mental issues')
plt.ylabel('Screen Time')
plt.title('Relationship between Screen Time and Mental Issues')
st.pyplot()

# Count plot for Feeling down, depressed or hopeless by Type of Usage
st.subheader("Count of Feeling down, depressed or hopeless by Type of Usage")
countplot_fig, countplot_ax = plt.subplots(figsize=(12, 8))
sns.countplot(x='Type of Usage', hue='Feeling down, depressed or hopeless', data=dataset)
plt.title('Count of Feeling down, depressed or hopeless by Type of Usage')
plt.xlabel('Type of Usage')
plt.xticks(rotation=60)
plt.ylabel('Count')
st.pyplot(countplot_fig)

# Bar plot for Relationship between Screen Time and Mental Fatigue
st.subheader("Relationship between Screen Time and Mental Fatigue")
barplot_fig, barplot_ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Screen Time', y='Feeling tired or having little energy', data=dataset, ci=None)  # ci=None to remove error bars
plt.title('Relationship between Screen Time and Mental Fatigue')
plt.xlabel('Screen Time')
plt.ylabel('Feeling tired or having little energy')
st.pyplot(barplot_fig)


# Define machine learning models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier()
}

# Machine Learning Model Section with styled title
st.sidebar.header("🤖 Machine Learning Model Evaluation 📊")

# Dropdown for selecting a machine learning model
selected_model = st.sidebar.selectbox("Select a Machine Learning Model", list(models.keys()))

# Train and evaluate the selected model
selected_model_instance = models[selected_model]
selected_model_instance.fit(X_train, y_train)
y_pred = selected_model_instance.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)

# Display accuracy with styled title
st.subheader(f"Accuracy for {selected_model}")
st.write(f"{accuracy:.3f}%")

# Display confusion matrix with styled title
st.subheader(f"Confusion Matrix for {selected_model}")
st.write(pd.DataFrame(cm, index=['No', 'Yes'], columns=['No', 'Yes']))

# Plot confusion matrix as a heatmap with styled title
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {selected_model}")
st.pyplot()


