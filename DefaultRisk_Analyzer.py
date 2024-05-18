from lime.lime_tabular import LimeTabularExplainer
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_echarts import st_echarts
from Functions.prep_functions import preprocess_data


############################ LOAD EXTERNAL OBJECTS ###############################

# Create LIME explainer.
X_train = pd.read_csv('./Objects/prestamos.csv')

# Crear la target en el conjunto original:
class_neg = ["Fully Paid", "Current", "In Grace Period", "Late (16-30 days)", "Does not meet the credit policy. Status:Fully Paid"]
class_pos = ["Charged Off", "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off", "Default"]
X_train['default'] = np.where(X_train['estado'].isin(class_pos), 1, 0)
vars_final = [
    'antigüedad_empleo_10+_years', 'antigüedad_empleo_2_years',
    'antigüedad_empleo_3_years', 'dti', 'finalidad_debt_consolidation',
    'finalidad_other', 'ingresos', 'ingresos_verificados_Not_Verified',
    'ingresos_verificados_Source_Verified', 'ingresos_verificados_Verified',
    'num_derogatorios', 'num_hipotecas', 'num_lineas_credito',
    'porc_tarjetas_75p', 'porc_uso_revolving', 'principal', 'rating_A',
    'rating_B', 'rating_C', 'rating_D', 'rating_E', 'rating_F-G',
    'tipo_interes', 'vivienda_MORTGAGE', 'vivienda_RENT'
]
X_train_prep = preprocess_data(X_train, vars_final)

model_final = joblib.load('./Objects/best_lgbmc_cw_trained_all_data')

############################ PAGE CONFIGURATION ###############################

st.set_page_config(
    layout='wide',
    page_title="DefaultRisk Analyzer",
    page_icon='./Images/page_icon.jpg',
    initial_sidebar_state="auto"
)

############################ SIDEBAR WIDGETS ###############################

with st.sidebar:
    st.title("*Client's Characteristics*")
    st.write('___')
    
    with st.form('characteristics'):
        col11, col12 = st.columns(2)
        col21, col22 = st.columns(2)
        col31, col32 = st.columns(2)
        col41, col42, col43= st.columns(3)
        col51, col52 = st.columns(2)
        col61, col62, col63 = st.columns(3)
        
        principal            = col11.number_input(label='**Pincipal ($)**', min_value=0.00)
        tipo_interes         = col12.number_input(label='**Annual Interest Rate ($)**', min_value=0.00, max_value=100.00)
        ingresos             = col21.number_input(label='**Annual Income ($)**', min_value=0.00)
        dti                  = col22.number_input(label='**Debit to Income ratio (%)**', min_value=0.00)
        porc_tarjetas_75p    = col31.number_input(label='**Percentage of credit cards with at least 75% usage**', min_value=0.00, max_value=100.00)
        porc_uso_revolving   = col32.number_input(label='**Percentage of revolving credit usage**', min_value=0.00)
        num_lineas_credito   = col41.number_input(label='**Number of credit lines**', min_value=0)
        num_hipotecas        = col42.number_input(label='**Number of mortgages**', min_value=0)
        num_derogatorios     = col43.number_input(label='**Number of derogatories**', min_value=0)
        antigüedad_empleo    = col51.selectbox(label='**Employment tenure**', options=['2 years', '3 years', '10+ years', 'other'])
        ingresos_verificados = col52.selectbox(label='**Verified income status**', options=['Not Verified', 'Verified', 'Source Verified'])
        rating               = col61.selectbox(label='**Rating**', options=['A', 'B', 'C', 'D', 'E', 'F-G'])
        finalidad            = col62.selectbox(label='**Loan purpose**', options=['debt consolidation', 'other'])
        vivienda             = col63.selectbox(label='**Housing type**', options=['MORTGAGE', 'RENT', 'other'])
        
        st.write('')
        submit = st.form_submit_button('Calculate Risk')
    
    # Get model values for categorical features:
    antigüedad_empleo_cats    = ['antigüedad_empleo_10+_years', 'antigüedad_empleo_2_years', 'antigüedad_empleo_3_years']
    ingresos_verificados_cats = ['ingresos_verificados_Not_Verified', 'ingresos_verificados_Source_Verified', 'ingresos_verificados_Verified']
    rating_cats               = ['rating_A', 'rating_B', 'rating_C', 'rating_D', 'rating_E', 'rating_F-G']
    finalidad_cats            = ['finalidad_debt_consolidation', 'finalidad_other']
    vivienda_cats             = ['vivienda_MORTGAGE', 'vivienda_RENT']
    
    antigüedad_empleo_output    = '_'.join(['antigüedad_empleo'] + antigüedad_empleo.split(' '))
    ingresos_verificados_output = '_'.join(['ingresos_verificados'] + ingresos_verificados.split(' '))
    rating_output               = '_'.join(['rating'] + rating.split(' '))
    finalidad_output            = '_'.join(['finalidad'] + finalidad.split(' '))
    vivienda_output             = '_'.join(['vivienda'] + vivienda.split(' '))
    
    cats_list   = []
    cats_list.append(antigüedad_empleo_cats)
    cats_list.append(ingresos_verificados_cats)
    cats_list.append(rating_cats)
    cats_list.append(finalidad_cats)
    cats_list.append(vivienda_cats)
    
    output_list = []
    output_list.append(antigüedad_empleo_output)
    output_list.append(ingresos_verificados_output)
    output_list.append(rating_output)
    output_list.append(finalidad_output)
    output_list.append(vivienda_output)
    
    cats_dict = {}
    for cats, output in zip(cats_list, output_list):
        for cat in cats:
            if output == cat:
                cats_dict[cat] = [1]
            else:
                cats_dict[cat] = [0]
    
    # Get model values for numerical features:
    values_dict ={
        'principal': [principal],
        'tipo_interes': [tipo_interes],
        'ingresos': [ingresos],
        'dti': [dti],
        'porc_tarjetas_75p': [porc_tarjetas_75p],
        'porc_uso_revolving': [porc_uso_revolving],
        'num_lineas_credito': [num_lineas_credito],
        'num_hipotecas': [num_hipotecas],
        'num_derogatorios': [num_derogatorios],
    }
    
    # Combine both dicts.
    values_dict.update(cats_dict)
    final_dict = dict(sorted(values_dict.items(), key=lambda x: x[0]))
    
    # Convert inputs into a single row DataFrame.
    client_df = pd.DataFrame(final_dict)
    
############################ INTERNAL CALCULATIONS ###############################

prob_prediction = model_final.predict_proba(client_df)[:, 1][0]
prob_prediction_perc = round(prob_prediction * 100, 2)

# Options for Velocímetro predicted prob for default.
veloc_options = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "Predicted default probability",
                "type": "gauge",
                "axisLine": {
                    "lineStyle": {
                        "width": 10,
                    },
                },
                "progress": {"show": "true", "width": 10},
                "detail": {"valueAnimation": "true", "formatter": "{value}"},
                "data": [{"value": prob_prediction_perc, "name": ""}],
            }
        ],
    }

# Explainability.
tab_explainer = LimeTabularExplainer(
    training_data=X_train_prep.values,
    mode='classification',
    feature_names=vars_final,
    random_state=0
)

instance_explained = tab_explainer.explain_instance(
    data_row=client_df.iloc[0],
    num_features=10,
    predict_fn=model_final.predict_proba,
    top_labels=None,
)

# df with prediction probabilities.
pred_probas = model_final.predict_proba(client_df)[0]
df_pred = pd.DataFrame(data=pred_probas).reset_index().rename(columns={0:'Prediction Probabilities', 'index':'Class'})
df_pred_styled = df_pred.style.apply(lambda row: ['background-color: green; color: white']*2 if row['Class'] == 0 else ['background-color: red; color: white']*2, axis=1)

# df with predictive features + LIME coefficients.
data_importance = instance_explained.as_list()
df_importance = pd.DataFrame(
    columns=['feature', 'coefficient'],
    data=data_importance
)

# Transform tree-based features into original features.
dict_importance_rules = dict(data_importance)
df_importance_features = pd.DataFrame(columns=['features', 'related coefficients'], data={var:v for k,v in dict_importance_rules.items() for var in vars_final if var in k}.items())

# client features + importance coefficients.
df_feat_val = client_df.T.reset_index().rename(columns={'index':'features', 0:'values'})
df_client_features = df_feat_val.merge(right=df_importance_features, how='inner', on='features').sort_values('related coefficients', key=lambda x: abs(x), ascending=False)

def highlight_coefficient(row):
    if row['related coefficients'] < 0:
        return ['background-color: green; color: white'] * len(row)
    else:
        return ['background-color: red; color: white'] * len(row)

# Apply the styling function to the DataFrame
df_client_features_styled = df_client_features.style.apply(highlight_coefficient, axis=1)

# barplot of local explanation.
colors = ['green' if val < 0 else 'red' for val in df_importance['coefficient']]

fig, ax = plt.subplots()
sns.barplot(data=df_importance, y='feature', x='coefficient', palette=colors, ax=ax)
ax.set_title('Local Explanation for Class 1 (Top 10 features)')
ax.set_xlabel('')
ax.set_ylabel('')

############################ BODY ###############################

# title
title1, title2 = st.columns(spec=[0.35, 0.65], gap='medium')
title1.image(
        image='./Images/sidebar_lupa.jpg',
        use_column_width='always'
    )
title2.write('')
title2.markdown("<h1 style=text-align: right; color: black; font-size: 50px;'>RISK SCORE ANALYZER</h1>", unsafe_allow_html=True)
title2.caption('Author: Hao, Qi Xu')
title2.caption('Source code: https://github.com/Haoqi9/CreditRisk_webapp_st/tree/master')
st.write('___')

# Guide
placeholder_init = st.empty()
placeholder_init.info("Please **enter the client's characteristics** and then click **'Calculate Risk'**")

# Display Output    
if submit is True:
    placeholder_init.empty()
    
    # Velocímetro display.
    st.markdown("<h1 style='text-align: center; color: grey; font-size: 34px;'>Predicted Default Probability</h1>", unsafe_allow_html=True)
    st_echarts(options=veloc_options, width="100%")
    st.write('___')
    
    # Prediction explainability.
    st.markdown("<h1 style='text-align: center; color: grey; font-size: 34px;'>Model Explainability</h1>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.write('___')
    
    st.markdown("<h1 style='text-align: center; color: grey; font-size: 34px;'>Details on the Prediction</h1>", unsafe_allow_html=True)
    col_b11, col_b12 = st.columns(spec=[0.3, 0.7], gap='large')
    col_b11.write(df_pred_styled)
    col_b12.write(df_client_features_styled)
