#  Inventory Analysis Web App

This is a **Streamlit-based web application** for analyzing inventory sales data.  
It allows users to upload an Excel (`.xlsx`) file and explore key insights such as product-wise sales, trends, and predictions.


# Features
- Upload your own **Excel file** for analysis
- Data cleaning and preprocessing
- Sales trend visualization
- Product-level insights
- Simple and interactive web interface


# Deployed Link
Click here to test the app 👉 [Inventory Analysis Web App](https://sudarshanverma19-inventoryanalysis-python-projectapp-zcmiwq.streamlit.app/)


# Dataset
The app requires a dataset to run.  
- Sample dataset is available here: [Sample Sales Data (Excel)](./data/sample_sales_data_for_forecasting_fixed.xlsx)  
- You can also access the **data folder** in the repository to get sample datasets.


# Installation & Usage
To run this project locally:

```bash
# Clone the repository
git clone https://github.com/sudarshanverma19/inventory-analysis-app.git

# Navigate into the folder
cd inventory-analysis-app

# Install required dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
