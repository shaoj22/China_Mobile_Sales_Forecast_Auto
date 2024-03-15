'''
File: main.py
Project: China_Mobile_Sales_Forecast_Auto
Description:
-----------
explain how to use the auto forecast.
-----------
Author: 626
Created Date: 2023-1107
'''


# to run notebook inside of package
import sys
sys.path.append('..')
# standard data manipulation imports
import pandas as pd
# import internal package functions
from src.plotting import *
from src.data_processing import *
from src.modeling import SalesForecasting


# 基本参数设定
value_col = 'sales' # name
diffed_value_col = f"{value_col}_differenced" # 标签name
date_col = 'date' # 日期name
mean_freq = 'Y'
forecast_horizon = 12 # test 周期
model_list = ['LinearRegression', 'RandomForest', 'XGBoost', 'LSTM', 'ARIMA'] # model name
# 原始数据读取
daily_data = pd.read_csv('train.csv')
# 月度数据处理
monthly_data = aggregate_by_time(
    daily_data, 
    date_col, 
    resample_freq='M', 
    aggregate='sum'
    )
print("月度数据：", monthly_data)
sales_list = monthly_data['sales'].tolist()
print("月度销售数据：", sales_list)
# 差分数据处理
monthly_data = difference_data(
    data=monthly_data, 
    date_col=date_col,   
    value_col=value_col, 
    diff_value_col_name=diffed_value_col
    )
print("差分数据：", monthly_data)
# 标签数据处理
supervised_data = create_lag_data(
    data=monthly_data, 
    date_col=date_col, 
    value_col=diffed_value_col, 
    lags=13)
print("标签数据：", supervised_data)
# train, test = create_train_test(supervised_data, test_size)

x_cols = list(supervised_data.drop(['store', 'item', date_col, value_col, diffed_value_col], axis=1).columns)
print("特征名称：", x_cols)
supervised_data = supervised_data[[date_col] + x_cols + [diffed_value_col]]
print(supervised_data)
train, test = create_train_test(supervised_data, forecast_horizon)
print(train)
print(test)
# 处理日期为index
scaler = DataScaler()
print('Train data shape: ', train.shape)
train_scaled = scaler.fit_transform(train).set_index(date_col)
test_scaled = scaler.transform(test).set_index(date_col)    
print('scaled train:')
print(train_scaled)
print('scaled test:')
print(test_scaled)
print('Train data shape: ', train_scaled.shape)
# 训练模型
model = SalesForecasting(model_list=model_list)
model.fit(train_scaled[x_cols], train_scaled[[diffed_value_col]])
# 预测测试
output = model.predict(test_scaled[x_cols], y_values=test_scaled[[diffed_value_col]], scaler=scaler)
# output = model.predict(test_scaled[x_cols], scaler=scaler)
print(output.stored_models['XGBoost']['predictions'])
diff_forecast_sales = output.stored_models['XGBoost']['predictions']
last_sale = sales_list[-13]
forecast_sales = []
for i in range(forecast_horizon):
    if i == 0:
        forecast_sales.append(last_sale + int(diff_forecast_sales[i]))
    else:
        forecast_sales.append(forecast_sales[-1] + int(diff_forecast_sales[i]))
print("真实值：", sales_list[-12:])
print("预测值：", forecast_sales)
results_plot = model.plot_results()
errors_plot = model.plot_errs()
output_df = pd.DataFrame(model.stored_models).T

plot_periodic_values_hist(daily_data, value_col)
plot_values_per_group(daily_data, value_col, ['store'])
plot_time_series(monthly_data, date_col, value_col, mean_freq)
# plot_time_series(monthly_data, date_col, diffed_value_col, mean_freq)
plt_acf_pcf(monthly_data, date_col, diffed_value_col)
plot_lag_cols(supervised_data, date_col, diffed_value_col, 'lag', num_lags=3)
visualize_train_test(train, test, date_col, diffed_value_col, figsize=(12,4))