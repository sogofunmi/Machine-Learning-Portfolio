import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def load_data(filepath):

    return pd.read_excel(filepath, sheet_name="E Comm")

def summary(df):
    data_summary = []
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_nulls = df[col_name].isnull().sum()
        num_non_nulls = df[col_name].notnull().sum()
        num_unique = df[col_name].nunique()

        data_summary.append({
            "col_name": col_name,
            "col_dtype": col_dtype,
            "num_nulls": num_nulls,
            "num_non_nulls": num_non_nulls,
            "num_unique": num_unique
        })

    return pd.DataFrame(data_summary)

def process_values(df):

    floats = df.select_dtypes(include=['float']).columns
    medians = df[floats].median()
    df[floats] = df[floats].fillna(medians)

    df = df.replace(to_replace="Mobile Phone", value="Phone")
    df = df.replace(to_replace=["CC", "COD"], value=["Credit Card", "Cash on Delivery"])
    df = df.replace(to_replace="COD", value="Cash on Delivery")

    return df

def data_split(df):

    full_train, df_test = train_test_split(df, test_size=0.2, random_state=3)
    df_train, df_val = train_test_split(full_train, test_size=0.25, random_state=3)

    y_train = df_train.Churn.values
    y_val = df_val.Churn.values
    y_test = df_test.Churn.values

    df_train = df_train.drop(columns=["Churn", "CustomerID"])
    df_val = df_val.drop(columns=["Churn", "CustomerID"])
    df_test = df_test.drop(columns=["Churn", "CustomerID"])

    return df_train, df_test, df_val, y_train, y_test, y_val

def encoding(df_train, df_val):

    dv = DictVectorizer(sparse=False)

    train_dicts = df_train.to_dict(orient="records")
    val_dicts = df_val.to_dict(orient="records")

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    return X_train, X_val, dv
