import requests

url = "http://localhost:8091"

customer = {'Tenure': 4.0,
 'PreferredLoginDevice': 'Computer',
 'CityTier': 3,
 'WarehouseToHome': 8.0,
 'PreferredPaymentMode': 'Cash on Delivery',
 'Gender': 'Female',
 'HourSpendOnApp': 6.0,
 'NumberOfDeviceRegistered': 2,
 'PreferedOrderCat': 'Others',
 'SatisfactionScore': 3,
 'MaritalStatus': 'Single',
 'NumberOfAddress': 4,
 'Complain': 1,
 'OrderAmountHikeFromlastYear': 15.0,
 'CouponUsed': 0.0,
 'OrderCount': 1.0,
 'DaySinceLastOrder': 0.0,
 'CashbackAmount': 120.9}


response = requests.post(url, json=customer).json()
print(response)

if response['Churn'] == True and (0.5 <= response['Probability'] <= 0.6):
    print("It might be best to send an email!")
elif response['Churn'] == True and (response['Probability'] > 0.6):
    print("Oh no! Send an email now!")

