# Required packages:
# Pandas
# XGBoost
# Scikit-learn
# Matplotlib
# Seaborn
# Streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

st.set_page_config(
    page_title = "Demand Forecast (Weeks 146-155)",
    page_icon = "Logo Placeholder.png",
    layout = "wide"
)

# Calculating MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100

# Loading and preprocessing training data
@st.cache_data
def load_training_data():
    trainCsvDf = pd.read_csv("train/train.csv")
    dosageInfoCsvDf = pd.read_csv("train/dosage_info.csv")
    plantInfoCsvDf = pd.read_csv("train/plant_info.csv")

    trainDf = pd.merge(trainCsvDf, dosageInfoCsvDf, on = 'drug_id', how = 'left')
    trainDf = pd.merge(trainDf, plantInfoCsvDf, on = 'plant_id', how = 'left')

    print(f"Number of duplicate rows: {trainDf.duplicated().sum()}")

    start_date = datetime(2021, 12, 26)

    # Calculate the date for each week
    trainDf['date'] = trainDf['week'].apply(lambda x: start_date + timedelta(weeks = x - 1))
    trainDf['month_of_year'] = trainDf['date'].dt.month
    trainDf['week_of_year'] = trainDf['date'].dt.isocalendar().week

    trainDf.sort_values(by = 'date', inplace = True)
    trainDf.reset_index(drop = True, inplace = True)

    trainDf['week_mod_52'] = trainDf['week'] % 52
    trainDf['discount'] = (trainDf['base_price'] > trainDf['wholesale_price']).astype(int)
    trainDf['markup'] = (trainDf['base_price'] < trainDf['wholesale_price']).astype(int)

    # Adding engineered features
    trainDf['demand_lag_1'] = trainDf.groupby(['plant_id', 'drug_id'])['num_orders'].shift(1)
    trainDf['demand_lag_2'] = trainDf.groupby(['plant_id', 'drug_id'])['num_orders'].shift(2)
    trainDf['demand_lag_4'] = trainDf.groupby(['plant_id', 'drug_id'])['num_orders'].shift(4)

    trainDf['demand_roll_mean_3'] = (
        trainDf.groupby(['plant_id', 'drug_id'])['num_orders']
        .rolling(3)
        .mean()
        .reset_index(level = [0, 1], drop = True)
    )

    trainDf['demand_roll_sum_5'] = (
        trainDf.groupby(['plant_id', 'drug_id'])['num_orders']
        .rolling(5)
        .sum()
        .reset_index(level = [0, 1], drop = True)
    )

    trainDf['promotion_active'] = (trainDf['medical_marketing_campaign'] | trainDf['homepage_featured']).astype(int)

    trainDf = pd.get_dummies(trainDf, columns = ['plant_scale', 'dosage_form', 'therapeutic_category'], drop_first = True)

    return trainDf

@st.cache_data
def load_visualizations_data():
    trainCsvDf = pd.read_csv("train/train.csv")
    dosageInfoCsvDf = pd.read_csv("train/dosage_info.csv")
    plantInfoCsvDf = pd.read_csv("train/plant_info.csv")

    vizDf = pd.merge(trainCsvDf, dosageInfoCsvDf, on = 'drug_id', how = 'left')
    vizDf = pd.merge(vizDf, plantInfoCsvDf, on = 'plant_id', how = 'left')

    return vizDf

# Train model
@st.cache_resource
def train_model(trainDf):
    categorical_dummies = [col for col in trainDf.columns if col.startswith('showroom_type_') 
                       or col.startswith('dosage_form_') 
                       or col.startswith('therapeutic_category_')]

    time_features = ['week_mod_52', 'month_of_year', 'week_of_year']

    features = (
        ['demand_lag_1', 'demand_lag_2', 'demand_lag_4',  # Lag features
         'demand_roll_mean_3', 'demand_roll_sum_5',       # Rolling features
         'discount', 'markup', 'promotion_active']        # Pricing and promo features
        + categorical_dummies                             # Categorical dummies
        + time_features                                   # Time-based features
    )

    # Training and validation split
    split_index = int(len(trainDf) * 0.8)
    
    # Sort the dataset by date or week to ensure sequential splitting
    trainDf_sorted = trainDf.sort_values(by = "date")
    
    # Perform the split
    train_data = trainDf_sorted.iloc[:split_index]
    val_data = trainDf_sorted.iloc[split_index:]

    X_train = train_data[features]
    y_train = train_data['num_orders']
    X_val = val_data[features]
    y_val = val_data['num_orders']

    xgb_model = XGBRegressor(
        objective = "reg:tweedie",
        n_estimators = 1575,      # Number of trees
        max_depth = 4,            # Maximum depth of trees
        learning_rate = 0.01,     # Step size shrinkage
        subsample = 1.0,          # Use all rows without subsampling
        colsample_bytree = 1.0,   # Use all features without subsampling
        colsample_bylevel = 1.0,  # Use all features at each level
        colsample_bynode = 1.0,   # Use all features at each split
        random_state = 8          # Ensures reproducibility
    )

    xgb_model.fit(X_train, y_train)

    y_val_pred = xgb_model.predict(X_val).round().astype(int)

    rmsle_val = np.sqrt(mean_squared_log_error(y_val, y_val_pred))
    print(f"RMSLE on Validation Set: {rmsle_val:.4f}")

    mape = calculate_mape(y_val, y_val_pred)
    print(f"MAPE: {mape:.2f}%")

    return xgb_model

# Load test data and make predictions
@st.cache_data
def predict_data(_xgb_model, trainDf):
    testXlsDf = pd.read_csv("test.xls")
    dosageInfoCsvDf = pd.read_csv("train/dosage_info.csv")
    plantInfoCsvDf = pd.read_csv("train/plant_info.csv")

    # Merge with dosage_info and plant_info
    testDf = pd.merge(testXlsDf, dosageInfoCsvDf, on = 'drug_id', how = 'left')
    testDf = pd.merge(testDf, plantInfoCsvDf, on = 'plant_id', how = 'left')

    # Preprocessing
    testDf['medical_marketing_campaign'] = testDf['medical_marketing_campaign'].astype(int)
    testDf['homepage_featured'] = testDf['homepage_featured'].astype(int)
    testDf['promotion_active'] = (testDf['medical_marketing_campaign'] | testDf['homepage_featured']).astype(int)

    # Perform dummy encoding for categorical variables
    testDf = pd.get_dummies(testDf, columns = ['plant_scale', 'dosage_form', 'therapeutic_category'], drop_first = True)

    # Add engineered features
    testDf['discount'] = (testDf['base_price'] > testDf['wholesale_price']).astype(int)
    testDf['markup'] = (testDf['base_price'] < testDf['wholesale_price']).astype(int)

    # 1. Append recent training data to test dataset
    recent_train_data = trainDf.sort_values(by = 'date').tail(10000)  # Last 10,000 rows of train data
    predicted_testDf = pd.concat([recent_train_data, testDf], axis = 0, ignore_index = True)

    # Add time-based features
    start_date = recent_train_data['date'].min() - timedelta(weeks = int(recent_train_data['week'].min() - 1))
    predicted_testDf['date'] = predicted_testDf['week'].apply(lambda x: start_date + timedelta(weeks = x - 1))

    predicted_testDf['month_of_year'] = predicted_testDf['date'].dt.month
    predicted_testDf['week_of_year'] = predicted_testDf['date'].dt.isocalendar().week
    predicted_testDf['week_mod_52'] = predicted_testDf['week'] % 52

    time_features = ['week_mod_52', 'month_of_year', 'week_of_year']

    categorical_dummies = [col for col in predicted_testDf.columns
                           if col.startswith('plant_scale_') or col.startswith('dosage_form_') or col.startswith('therapeutic_category_')]

    features = ([
        'demand_lag_1', 'demand_lag_2', 'demand_lag_4',  # Lag features
        'demand_roll_mean_3', 'demand_roll_sum_5',       # Rolling features
        'discount', 'markup', 'promotion_active']        # Pricing and promo features
        + categorical_dummies
        + time_features
    )

    for col in ['demand_lag_1', 'demand_lag_2', 'demand_lag_4', 'demand_roll_mean_3', 'demand_roll_sum_5']:
        predicted_testDf[col] = 0

    # 2. Autoregressing lag features
    predictions = []
    for idx in range(len(testDf)):
        current_row = predicted_testDf.iloc[len(recent_train_data) + idx]

        # Prepare features for the current prediction
        X_test_row = current_row[_xgb_model.get_booster().feature_names].values.reshape(1, -1)

        # Predict `num_orders`
        predicted_orders = _xgb_model.predict(X_test_row).round().astype(int)[0]
        predictions.append(predicted_orders)

        # Update the current row with the prediction
        predicted_testDf.loc[len(recent_train_data) + idx, 'num_orders'] = predicted_orders

        # Update lag and rolling features for the next rows
        predicted_testDf['demand_roll_mean_3'] = 0.0
        predicted_testDf['demand_roll_sum_5'] = 0.0

        if idx + 1 < len(testDf):
            next_idx = len(recent_train_data) + idx + 1
            # Update lag features
            predicted_testDf.loc[next_idx, 'demand_lag_1'] = predicted_orders
            predicted_testDf.loc[next_idx, 'demand_lag_2'] = predicted_testDf.loc[next_idx - 1, 'demand_lag_1']
            predicted_testDf.loc[next_idx, 'demand_lag_4'] = predicted_testDf.loc[next_idx - 1, 'demand_lag_2']

            # Update rolling features
            recent_predictions = predictions[max(0, idx - 2):idx + 1]
            predicted_testDf.loc[next_idx, 'demand_roll_mean_3'] = np.mean(recent_predictions)
            predicted_testDf.loc[next_idx, 'demand_roll_sum_5'] = np.sum(predictions[max(0, idx - 4):idx + 1])

    # Ensure order is preserved by returning only the predicted rows
    predicted_testDf = predicted_testDf.iloc[len(recent_train_data):].reset_index(drop=True)

    # Save predictions
    submission = predicted_testDf[['id', 'num_orders']]
    predicted_testDf.to_csv('submission.csv', index = False)

    print("Predictions completed and saved to 'submission.csv'")
    return predicted_testDf

trainingDf = load_training_data()
xgb_model = train_model(trainingDf)
predictedDf = predict_data(xgb_model, trainingDf)

### S T R E A M L I T ###

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Demand Forecast"])

if page == "Dashboard":
    st.title("Dashboard")
    st.subheader("Reports")

    vizDf = load_visualizations_data()

    sns.set_theme(style = "whitegrid")

    col1, col2 = st.columns([2, 2], gap = "large")

    with col1:
        # 1. Weekly Demand Seasonality
        fig1 = plt.figure(figsize = (8, 6))
        sns.lineplot(data = vizDf.groupby('week', as_index = False)['num_orders'].sum(),
                     x = 'week', y = 'num_orders', errorbar = None)
        plt.title("Demand Trends Over Weeks")
        plt.xlabel("Weeks")
        plt.ylabel("Number of Orders")
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown("<br>", unsafe_allow_html = True)

    with col2:
        # 2. Weekly Demand Broken Down by Dosage Form 
        fig2 = plt.figure(figsize = (8, 6))
        sns.lineplot(data = vizDf.groupby(['week', 'dosage_form'], as_index = False)['num_orders'].sum(),
                     x = 'week', y = 'num_orders', hue = 'dosage_form', palette = "viridis", errorbar = None)
        plt.legend(title = "Dosage Form", loc = 'upper right')
        plt.title("Demand Trends by Dosage Form")
        plt.xlabel("Weeks")
        plt.ylabel("Number of Orders")
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown("<br>", unsafe_allow_html = True)

    st.markdown("Tablets and Syrups are the most popular dosage forms. These two dosage forms are responsible for more than 50% of the demand.<br>Tablets, Syrups, Capsules, and Ointments together account for more than 75% of the demand.", unsafe_allow_html = True)

    col1, col2 = st.columns([2, 2], gap = "large")

    with col1:
        # 3. Total Demand by Plant Scale Type
        fig3 = plt.figure(figsize = (8, 6))
        sns.barplot(data = vizDf, x = 'plant_scale', y = 'num_orders', estimator = sum, hue = 'plant_scale', palette = "viridis", errorbar = None)
        plt.title("Total Demand by Plant Scale Type")
        plt.xlabel("Plant Scale Type")
        plt.ylabel("Total Number of Orders")
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown("<br>", unsafe_allow_html = True)

    with col2:
        # 4. Average Demand by Plant Scale Type
        fig4 = plt.figure(figsize = (8, 6))
        sns.barplot(data = vizDf, x = 'plant_scale', y = 'num_orders', hue = 'plant_scale', palette = "viridis", errorbar = None)
        plt.title("Average Demand by Plant Scale Type")
        plt.xlabel("Plant Scale Type")
        plt.ylabel("Average Number of Orders")
        plt.tight_layout()
        st.pyplot(fig4)
        st.markdown("<br>", unsafe_allow_html = True)

    st.write("Standard Production gets more orders in total, but Large Scale Production gets larger orders on average.")

    col1, col2 = st.columns([2, 2], gap = "large")

    with col1:
        # 5. Average Demand by therapeutic_category
        fig5 = plt.figure(figsize = (8, 6))
        sns.barplot(data = vizDf, x = 'therapeutic_category', y = 'num_orders', hue = 'therapeutic_category', palette = "viridis", errorbar = None)
        #plt.xticks(rotation = 45)
        plt.xlabel("Therapeutic Category")
        plt.ylabel("Average Number of Orders")
        plt.title("Average Demand by Therapeutic Category")
        plt.tight_layout()
        st.pyplot(fig5)
        st.markdown("<br>", unsafe_allow_html = True)

    with col2:
        # 5. Average Demand by Therapeutic Category and Center Type
        fig6 = plt.figure(figsize = (8, 6))
        therapeutic_category_plant_scale_orders = vizDf.groupby(['plant_scale', 'therapeutic_category'], as_index = False)['num_orders'].mean()
        sns.barplot(data = therapeutic_category_plant_scale_orders, x = 'plant_scale', y = 'num_orders', hue = 'therapeutic_category', palette = "viridis")
        plt.title("Average Demand by Therapeutic Category and Plant Scale Type")
        plt.xlabel("Center Type")
        plt.ylabel("Average Number of Orders")
        plt.legend(title = "Therapeutic Category")
        plt.tight_layout()
        st.pyplot(fig6)
        st.markdown("<br>", unsafe_allow_html = True)

    st.write("Antibiotics are the most popular. However, in Large Scale plants, Pain Relief medication is also highly sought after.")

    col1, col2 = st.columns([2, 2], gap = "large")

    with col1:
        # 7. Scatterplot: Price vs Demand
        fig7 = plt.figure(figsize = (8, 6))
        sns.scatterplot(data = vizDf, x = 'wholesale_price', y = 'num_orders', hue = 'wholesale_price', palette = "viridis", legend = False, alpha = 0.3)
        plt.title("Checkout Price vs Demand")
        plt.xlabel("Checkout Price")
        plt.ylabel("Number of Orders")
        plt.tight_layout()
        st.pyplot(fig7)
        st.markdown("<br>", unsafe_allow_html = True)

    with col2:
        # 8. Impact of Promotions on Demand
        fig8 = plt.figure(figsize = (8, 6))
        sns.boxplot(data = vizDf, x = 'medical_marketing_campaign', y = 'num_orders', hue = 'medical_marketing_campaign', palette = "viridis", hue_order = [0, 1])
        plt.xticks([0, 1], ['No Promotion', 'Promotion'])
        legend_labels = ['No Promotion', 'Promotion']
        plt.legend(title = "Promotion Status", labels = legend_labels, loc = 'upper right')
        plt.title("Impact of Promotions on Demand")
        plt.xlabel(None)
        plt.ylabel("Orders")
        plt.tight_layout()
        st.pyplot(fig8)
        st.markdown("<br>", unsafe_allow_html = True)

    st.write("For orders above $500, the density is significantly low. Promotions targeting this segment could help drive demand and improve sales.")

elif page == "Demand Forecast":
    st.title("Staff and Raw Ingredient Calculator")

    st.subheader("Demand graphs for each plant type")
    col1, col2, col3 = st.columns(3, gap = "large")

    # Plot Demand Trends for Standard_Production
    with col1:
        #plant_scale_Standard_Production_data = predictedDf[(predictedDf['plant_scale_Large_Scale'] == 0) & (predictedDf['plant_scale_Small_Batch'] == 0)]
        plant_scale_Standard_Production_data = predictedDf[(predictedDf['plant_scale_Standard_Production'] == 1)]
        aggregated_Standard_Production_data = plant_scale_Standard_Production_data.groupby('week', as_index = False)['num_orders'].sum()

        st.subheader("Standard Production") # Type A
        plt.figure(figsize = (6, 4))
        sns.lineplot(
            data = aggregated_Standard_Production_data,
            x = "week",
            y = "num_orders",
            errorbar = None
        )
        plt.xlim(146, 156)
        plt.title("Demand Trends for Standard_Production_Plants")
        plt.xlabel("Week")
        plt.ylabel("Number of Orders")
        st.pyplot(plt)

    # Plot Demand Trends for Large_Scale
    with col2:
        #plant_scale_Large_Scale_data = predictedDf[(predictedDf['plant_scale_Large_Scale'] == 1)]
        plant_scale_Large_Scale_data = predictedDf[(predictedDf['plant_scale_Standard_Production'] == 0) & (predictedDf['plant_scale_Small_Batch'] == 0)]
        aggregated_Large_Scale_data = plant_scale_Large_Scale_data.groupby('week', as_index = False)['num_orders'].sum()

        st.subheader("Large Scale") # Type B
        plt.figure(figsize = (6, 4))
        sns.lineplot(
            data = aggregated_Large_Scale_data,
            x = "week",
            y = "num_orders",
            errorbar = None
        )
        plt.xlim(146, 156)
        plt.title("Demand Trends for Large_Scale_Plants")
        plt.xlabel("Week")
        plt.ylabel("Number of Orders")
        st.pyplot(plt)

    # Plot Demand Trends for Small_Batch
    with col3:
        plant_scale_Small_Batch_data = predictedDf[(predictedDf['plant_scale_Small_Batch'] == 1)]
        aggregated_Small_Batch_data = plant_scale_Small_Batch_data.groupby('week', as_index = False)['num_orders'].sum()

        st.subheader("Small Batch") # Type C
        plt.figure(figsize = (6, 4))
        sns.lineplot(
            data = aggregated_Small_Batch_data,
            x = "week",
            y = "num_orders",
            errorbar = None
        )
        plt.xlim(146, 156)
        plt.title("Demand Trends for Small_Batch_Plants")
        plt.xlabel("Week")
        plt.ylabel("Number of Orders")
        st.pyplot(plt)
    
    st.write("Demand forecasts")
    st.subheader("Inventory and Staffing Requirements for Weeks 146 to 155")

    st.write("Please input the number of man-hours required to prepare **1 order** of each medicine type")

    # Input fields for each Therapeutic Category
    col1, col2, col3, col4 = st.columns(4, gap = "large")
    
    with col1:
        pain_relief = st.slider("Pain Relief Medication", min_value = 1, max_value = 20, value = 3, step = 1)
    
    with col2:
        cardiovascular = st.slider("Cardiovascular Medication", min_value = 1, max_value = 20, value = 4, step = 1)
    
    with col3:
        antibiotics = st.slider("Antibiotic Medication", min_value = 1, max_value = 20, value = 2, step = 1)
    
    with col4:
        respiratory = st.slider("Respiratory Medication", min_value = 1, max_value = 20, value = 1, step = 1)

    # Merge staff inputs into a dictionary for mapping
    staff_inputs = {
        'Pain Relief': pain_relief,
        'Cardiovascular': cardiovascular,
        'Antibiotics': antibiotics,
        'Respiratory': respiratory
    }

    # Calculate staff required for each Therapeutic Category directly
    def calculate_staff(row):
        if row['therapeutic_category_Pain Relief'] == 1:
            return row['num_orders'] * staff_inputs['Pain Relief']
        elif row['therapeutic_category_Cardiovascular'] == 1:
            return row['num_orders'] * staff_inputs['Cardiovascular']
        elif row['therapeutic_category_Respiratory'] == 1:
            return row['num_orders'] * staff_inputs['Respiratory']
        else:
            return row['num_orders'] * staff_inputs['Antibiotics']

    # Filter and prepare data for table creation
    predictedDf['staff_required'] = predictedDf.apply(calculate_staff, axis = 1)

    # Inventory per food Dosage Form
    def calculate_inventory(row):
        inventory = 0
        if row.filter(like = 'dosage_form_').sum() == 0:
            inventory += row['num_orders'] * inventory_per_dosage_form['Tablets']
    
        # Add inventory for all other dosage forms
        for dosage_form in dosage_forms:
            if dosage_form != "Tablets":  # Skipping hinge dosage_form
                col_name = f"dosage_form_{dosage_form}"
                if col_name in row and row[col_name] == True:
                    inventory += row['num_orders'] * inventory_per_dosage_form[dosage_form]
    
        return inventory

    st.write("Please input the units of inventory required to prepare **1 order** of each dosage form")

    # Input fields for each Therapeutic Category
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7, gap = "large")

    with col1:
        inhaler_dosage_form = st.number_input("Inhaler", min_value = 1, value = 5, step = 1)
        solution_dosage_form = st.number_input("Solution", min_value = 1, value = 2, step = 1)

    with col2:
        injection_dosage_form = st.number_input("Injection", min_value = 1, value = 6, step = 1)
        gel_dosage_form = st.number_input("Gels", min_value = 1, value = 3, step = 1)

    with col3:
        lozenge_dosage_form = st.number_input("Lozenges", min_value = 1, value = 2, step = 1)
        drops_dosage_form = st.number_input("Drops", min_value = 1, value = 2, step = 1)

    with col4:
        syrup_dosage_form = st.number_input("Syrups", min_value = 1, value = 3, step = 1)
        Capsule_dosage_form = st.number_input("Capsules", min_value = 1, value = 2, step = 1)

    with col5:
        Ointment_dosage_form = st.number_input("Ointments", min_value = 1, value = 2, step = 1)
        powder_dosage_form = st.number_input("Powders", min_value = 1, value = 1, step = 1)

    with col6:
        implant_dosage_form = st.number_input("Implants", min_value = 1, value = 1, step = 1)
        patches_dosage_form = st.number_input("Patches", min_value = 1, value = 1, step = 1)

    with col7:
        Tablet_dosage_form = st.number_input("Tablets", min_value = 1, value = 2, step = 1)
        extra_dosage_form = st.number_input("Extras", min_value = 1, value = 1, step = 1)

    dosage_forms = [
        "Tablets", "Extras", "Powder", "Patches", "Ointment", "Syrup", "Drops", 
        "Capsule", "Lozenge", "Implant", "Injection", "Gel", "Solution", "Inhaler"
    ]

    inventory_per_dosage_form = {
        'Inhaler': inhaler_dosage_form,
        'Solution': solution_dosage_form,
        'Injection': injection_dosage_form,
        'Gel': gel_dosage_form,
        'Lozenge': lozenge_dosage_form,
        'Drops': drops_dosage_form,
        'Syrup': syrup_dosage_form,
        'Capsule': Capsule_dosage_form,
        'Ointment': Ointment_dosage_form,
        'Powder': powder_dosage_form,
        'Implant': implant_dosage_form,
        'Patches': patches_dosage_form,
        'Tablets': Tablet_dosage_form,
        'Extras': extra_dosage_form
    }

    predictedDf['inventory_required'] = predictedDf.apply(calculate_inventory, axis = 1)

    inventory_staff_table = (
        predictedDf
        .groupby(['week', 'date'], as_index = False)
        .agg({
            'date': 'first',
            'num_orders': 'sum',
            'inventory_required': 'sum',
            'staff_required': 'sum'
        })
    )

    inventory_staff_table['date'] = inventory_staff_table['date'].dt.strftime('%m-%d-%Y')

    inventory_staff_table.rename(columns = {
        'week': 'Week',
        'date': 'Date',
        'num_orders': 'Total Orders',
        'staff_required': 'Total Staff Required',
        'inventory_required': 'Inventory Count'
    }, inplace = True)

    total_staff = inventory_staff_table['Total Staff Required'].sum().astype(int)
    total_inventory = inventory_staff_table['Inventory Count'].sum().astype(int)

    # Add metrics to the bottom of the sidebar
    with st.sidebar:
        st.markdown("---") 
        st.markdown("<h3>Summary Metrics<br>For The Next 10 Weeks</h3>", unsafe_allow_html=True)
        st.metric(label = "Total Inventory Count", value = f"{total_inventory:,}")
        st.metric(label = "Total Man-hours Required", value = f"{total_staff:,}")

    st.markdown("---")
    st.dataframe(inventory_staff_table, use_container_width = True)
    #col1, col2 = st.columns([3, 1])

    #with col1:
    #    st.dataframe(ingredients_staff_table)

    #with col2:
    #    st.markdown("<br><br>", unsafe_allow_html = True)
    #    st.image("Logo Placeholder.png", use_container_width = True)