# SecondDrive: Used Car Price Prediction & Recommendation System 🚗💨
## Project Overview
Welcome to SecondDrive, an intelligent web application designed to help users navigate the used car market with confidence. Leveraging machine learning, SecondDrive provides two core functionalities: estimating the resale price of a used car based on its specifications and recommending cars from a dataset that match a user's budget and preferences. Whether you're looking to sell your car at a fair price or buy one that fits your needs, SecondDrive is your reliable companion.

## Features ✨
SecondDrive offers a seamless experience through its two distinct modes:

## Check Price Mode 💰:

* Input various car details such as brand, manufacturing year, engine capacity (CC), mileage, fuel type, transmission, and more.

* Get an instant, data-driven estimated resale price for the car.

* Helps sellers set competitive prices and buyers understand market value.

## Suggest Car Mode 🎯:

* Define your budget range, preferred mileage, and other relevant filters.

* Receive tailored recommendations of cars from the dataset that best align with your criteria.

* Simplifies the car buying process by filtering through numerous options.

## Dataset Description 📊
The machine learning model powering SecondDrive was trained on a comprehensive and thoroughly cleaned dataset of used cars. Key aspects of the data handling include:

* Cleaning: Extensive preprocessing was performed to handle missing values, outliers, and inconsistencies.

* Categorical Features: Features like brand, fuel type, and transmission were transformed using one-hot encoding.

* Numeric Features: Numerical features such as engine CC and mileage were scaled to ensure optimal model performance.

* Target Transformation: The target variable, price_usd, was log-transformed (log(price_usd)) to mitigate skewness and improve the model's predictive accuracy.

## Tech Stack 🛠️
* Python: The core programming language for data processing, model training, and application logic.

* Pandas: For data manipulation and analysis.

* NumPy: For numerical operations.

* Scikit-learn: For machine learning model training, preprocessing, and evaluation.

* Streamlit: Used for creating the interactive and user-friendly web interface.

* Joblib: Employed to efficiently save and load the trained machine learning model and the feature scaler, ensuring quick deployment.

## How to Run the App Locally 🚀
Follow these simple steps to get SecondDrive up and running on your local machine:

* Clone the Repository:
```
git clone https://github.com/your-username/SecondDrive.git
cd SecondDrive
```
(Replace your-username/SecondDrive.git with the actual repository URL)

* Create a Virtual Environment (Recommended):
```
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
* Install Dependencies:
```
pip install -r requirements.txt
```
(Ensure you have a requirements.txt file listing all necessary packages like pandas, numpy, scikit-learn, streamlit, joblib)

* Prepare Model and Scaler Files:

Ensure your trained model (model.joblib) and scaler (scaler.joblib) files are present in the project directory, typically in a models/ or artifacts/ folder.

(If these files are not committed to the repo, you'll need to run the model training script first to generate them.)

* Run the Streamlit Application:
```
streamlit run main.py
```
(Assuming your main Streamlit application file is named app.py)

* Access the App:

After running the command, Streamlit will open the application in your default web browser (e.g., http://localhost:8501).

## Model Training Workflow ⚙️
The machine learning model in SecondDrive undergoes a robust training process:

* Data Loading: The raw used car dataset is loaded.

* Data Preprocessing:

    * Missing values are handled (e.g., imputation or removal).

Outliers are identified and treated.

Irrelevant features are dropped.

* Feature Engineering:

Categorical features are identified.

One-hot encoding is applied to convert categorical data into a numerical format suitable for machine learning algorithms.

* Target Transformation:

The price_usd target variable is log-transformed using np.log1p() to reduce skewness and stabilize variance.

* Data Splitting: The dataset is split into training and testing sets to evaluate model performance on unseen data.

* Feature Scaling: Numeric features in the training data are scaled (e.g., using StandardScaler or MinMaxScaler) to bring them to a similar range. The scaler is then fitted on the training data and used to transform both training and testing data.

* Model Selection & Training: A suitable regression model (e.g., Linear Regression, RandomForest Regressor, Gradient Boosting) is chosen and trained on the preprocessed and scaled training data.

* Model Evaluation: The trained model's performance is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), or R-squared on the test set.

* Model Persistence: The trained model and the fitted scaler are saved using joblib for later use in the Streamlit application.

## Future Improvements 🌟
I am continuously working to enhance SecondDrive. Here are some planned future improvements:

* Advanced Recommendation Algorithms: Implement more sophisticated recommendation engines (e.g., collaborative filtering) for better car suggestions.

* Integration with Live Data: Explore options to fetch real-time or more frequently updated used car listings.

* User Authentication: Add user accounts to save preferences, search history, and favorite cars.

* Deployment: Deploy the application to a cloud platform (e.g., Heroku, AWS, Google Cloud) for wider accessibility.

* Model Retraining Pipeline: Automate the model retraining process with new data periodically.

* Interactive Visualizations: Include charts and graphs to show price trends or feature importance.

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

## Author ✍🏻
Sk Akib Ahammed[ahammedskakib@gmail.com]
