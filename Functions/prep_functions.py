import pandas as pd

# Preprocessing functions:
def impute_missing(df_train):
    # Evitar sobreescribir variable.
    df_train = df_train.copy()

    # Eliminar registros del conjunto de entrenamiento.
    mask_zero_notVerified = (df_train.ingresos == 0) & (df_train.ingresos_verificados == 'Not Verified')
    df_train = df_train.loc[~mask_zero_notVerified] 
    
    mask_dbt_neg = df_train.dti < 0
    df_train = df_train[~mask_dbt_neg]    
    
    rare_cats = ["ANY", "OTHER", "NONE"]
    mask_vivienda_rare = df_train.vivienda.isin(rare_cats)
    df_train = df_train[~mask_vivienda_rare]

    # Partición X_train / y_train.
    X_train = df_train.drop(columns='default')

    # Imputación por la moda.
    vars_moda = ["antigüedad_empleo", "num_hipotecas", "num_derogatorios"]  
    for var in vars_moda:
        X_train[var].fillna(X_train[var].mode().iloc[0], inplace=True)

    # Imputación por la mediana.
    vars_mediana = ["porc_uso_revolving", "dti", "num_lineas_credito", "porc_tarjetas_75p"]
    for var in vars_mediana:
        X_train[var].fillna(X_train[var].median(), inplace=True)

    return X_train
    

def get_derived_features(X):
    # Evitar sobreescribir variable.
    X = X.copy()

    # Agrupar categorías infrecuentes.
    X["rating"].replace(
        to_replace=["F", "G"],
        value="F-G",
        inplace=True
    )
    
    X["finalidad"].replace(
        to_replace=["major_purchase", "medical", "small_business", "car", "moving", "vacation", "house", "wedding", "renewable_energy", "educational"],
        value="other",
        inplace=True
    )
    
    # OHE.
    X = pd.get_dummies(
        data=X,
        columns=['antigüedad_empleo', 'finalidad', 'ingresos_verificados', 'rating', 'vivienda'],
        drop_first=False,
        dtype=float
    )

    return X


def preprocess_data(df_train, vars_final):
    # Evitar sobreescribir.
    df_train = df_train.copy()

    # Imputar nulos.
    X_train = impute_missing(df_train)

    # Feature engineering.
    X_train = get_derived_features(X_train)

    # Cambiar nombre de categorías acorde al output generado por LGBM.
    X_train.rename(columns={
        'num_cuotas_ 36 months': 'num_cuotas__36_months',
        'antigüedad_empleo_10+ years': 'antigüedad_empleo_10+_years',
        'antigüedad_empleo_2 years': 'antigüedad_empleo_2_years',
        'antigüedad_empleo_3 years': 'antigüedad_empleo_3_years',
        'ingresos_verificados_Not Verified': 'ingresos_verificados_Not_Verified',
        'ingresos_verificados_Source Verified': 'ingresos_verificados_Source_Verified'
    }, inplace=True)
    
    # Limitar a las variables finales.
    X_train_prep  = X_train[vars_final]

    return X_train_prep