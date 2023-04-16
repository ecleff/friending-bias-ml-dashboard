import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time
import streamlit_theme as stt



# from streamlit import configuration
# configuration.set_config_file(config_file='config.toml')
# 
sc = pd.read_csv("data/friending_bias_viz_data.csv", dtype={'ncessch': str,'high_school': str})
df_rf_sub = pd.read_csv("data/std_knn_rf_df.csv")

# setting color theme
pc_theme = {
 "primaryColor": "#972791",
    "backgroundColor": "#f5f5f5",
    "secondaryBackgroundColor": "#e6e6e6",
    "textColor": "#333333",
    "font": "sans-serif"
}


st.set_page_config(page_title="Predicting Friending Bias", page_icon=":sunglasses:", layout="wide")

# title
st.title('Exploring Predictive Models for Friending Bias')
st.caption("Emily Leff, Spring 2023")
st.header('Purpose')
st.markdown("This project investigates variables predicting friending bias to support future research optimizing rezoning for friendships, as opposed to or in addition to diversity. The end goal of rezoning school districts to optimize for diversity is to increase opportunities for all students through providing more avenues for students to increase their social capital. That only happens if diverse student groups are genuinely forming connections with one another.")
st.markdown("Before running rezoning for friending bias however, we wanted to confirm if friending bias could be taken as given for a school/district/block, or if there were demographic characteristics that directly predicted friending bias on the school level.")
st.header('What is friending bias?')
st.markdown("Friending bias is defined as 'the likelihood that low-income people form friendships with the high-income (above-median) income people that they're exposed to.' Friending bias is derived from Raj Chetty et al's Opportunity Index report, which draws from privacy-protected data on 21 billion friendships from Facebook.")
st.markdown("In context: A school with an average friending bias closer to -1 means that people at that school have a lower friending bias, meaning that the low-income students are more likely to form friendships with the high-income students they're exposed to.")
st.markdown("Likewise, a school with a friending bias closer to 1 means that people at that school have a higher friending bias, meaning these cross-SES friendships are less likely to occur.")

st.header("Findings")
st.markdown("We found that random forest and SVM regression were the models that yielded the highest predictive power. However, even these R-squared values were <0.4, and we concluded that they were not sufficient to strongly predict friending bias and therefore, friending bias should be taken as given for future rezoning purposes.")
st.markdown("For the SVM regressor, clustering and parental education attainment were the features of highest importance. For random forest, the most important features were clustering, volunteering rates, and the population of white students at the school. Below, users can explore variable feature importances for the random forest model in different data contexts.")

st.header('Data')
st.markdown("To begin to analyze factors predicting friending bias in U.S. public schools, we identified demographic features on the block and block-group level that pertain to student outcomes, in addition to the block level demographic information from the social capital dataset. The added Census data includes variables taken from the Chicago Public School's SES diversity criteria, including (1) median family income, (2) adult educational attainment, (3) the percentage of single-family households, (4) home-ownership percentage, & (5) percentage of population that speaks a language other than English. Other Census variables were included to supplement the definition of socioeconomic diversity, including public assistance status, median gross rent as a percentage of household income, average commute time for household head, and poverty-income ratio. Data files can be found here: https://drive.google.com/drive/folders/1dDoTyjGl5AHuoaCb591WGle2Mz-MvULF?usp=sharing.")
with st.expander("Full codebook for analysis"):
    st.write("From Social Capital Atlas")
    st.write("*Clustering: the rate with which two friends of a given person are in turn friends with each other.")
    st.write("*Volunteering rate: the share of people who are members of volunteering groups")
    st.write("From the U.S. Census Bureau:")
    st.write("*Means of Transportation to Work by Travel Time to Work Universe: Workers 16 years and over who did not work from home Source code: B08134 NHGIS code: AMQH")
    st.write("*Public Assistance Income or Food Stamps/SNAP in the Past 12 Months for Households Universe: Households Source code: B19058 NHGIS code: AMST")
    st.write(" *Median Family Income in the Past 12 Months (in 2020 Inflation-Adjusted Dollars) Universe: Families Source code: B19113 NHGIS code: AMS6")
    st.write("*Median Gross Rent as a Percentage of Household Income in the Past 12 Months (Dollars) Universe: Renter-occupied housing units paying cash rent Source code: B25071 NHGIS code: AMV6") 
    st.write("*Ratio of Income to Poverty Level in the Past 12 Months Universe: Population for whom poverty status is determined Source code: C17002 NHGIS code: AMZM")
    st.write("From CPS SES indicators:")
    st.write("*Household Type (Including Living Alone) Universe: Households Source code: B11001 NHGIS code: AOO1 ")
    st.write("*Educational Attainment for the Population 25 Years and Over Universe: Population 25 years and over Source code: B15003 NHGIS code: AOP8")
    st.write("*Household Language by Household Limited English Speaking Status Universe: Households Source code: C16002 NHGIS code: AOXV ")
    st.write("*Tenure Universe: Occupied housing units Source code: B25003 NHGIS code: AOSP")
    st.write("From GreatSchools.org")
    st.write("*AP Courses")
    st.write("*Honors Classes (binary)")
    st.write("*Great Schools Overall Score: summary snapshot of school quality based on test scores, student progress, and equity.")
    st.write("*Great Schools Equity Score: how well this school is serving disadvantaged students relative to all students, compared to other schools in the state, based on college readiness metrics, student progress, and test scores provided from the state’s Department of Education.")
    st.write("*Great Schools Test Scores: annual state test results for this school compared with scores statewide.")

    # fix formatting

st.divider()
# Variable scatterplots

#  transform data wide to long
densplot = pd.melt(sc, id_vars=['ncessch','bias_own_ses_hs',],
value_vars=[
'med_family_income', 'rent_pctage_household_income',
'num_white','num_black','num_native','num_asian', 'num_hispanic', 'num_ell',
'pct_commute_less_than_29','pct_commute_greater_than_30','pct_snap_no_cash_public_assist',
'pct_snap_with_cash_public_assist', 'pct_poverty_inc_ratio_185_under', 'ap_courses','honors_classes','gs_equity_rating',
'gs_overall_rating','gs_test_rating','gs_progress_rating','clustering_hs','volunteering_rate_hs','pct_married_couple_families'
, 'pct_single_parent_families' ,  'pct_non_english_lang' , 'pct_homeowners' , 'pct_less_than_hs_edu', 'pct_hs_degree', 'pct_associates_degree',
'pct_bachelors_degree', 'pct_post_grad_higher_degree'
], var_name="Variable", value_name="Value")



# subsets
# CPS variables - interactive scatterplot
cps = ['pct_homeowners', 'pct_less_than_hs_edu', 'pct_hs_degree', 'pct_associates_degree',
'pct_bachelors_degree', 'pct_post_grad_higher_degree', 'pct_single_parent_families' ,  'pct_non_english_lang',
'med_family_income']

cps_scatter = densplot[densplot['Variable'].isin(cps)]

input_dropdown_cps = alt.binding_select(options=['pct_homeowners', 'pct_less_than_hs_edu', 'pct_hs_degree', 'pct_associates_degree',
'pct_bachelors_degree', 'pct_post_grad_higher_degree', 'pct_single_parent_families' ,  'pct_non_english_lang',
'med_family_income'], name='CPS Factors')
selection_cps = alt.selection_single(fields=['Variable'], bind=input_dropdown_cps)

cps_scatter_chart = alt.Chart(cps_scatter, title="Chicago Public Schools (CPS) SES Diversity Criteria").mark_point(color="white").encode(
    x='Value:Q',
    y='bias_own_ses_hs:Q',
    tooltip='Variable:N'
).add_selection(
    selection_cps
).transform_filter(
    selection_cps
)
  

# Social Capital variables

# scatter plot
social_cap = ['clustering_hs', 'volunteering_rate_hs']
social_cap_scatter = densplot[densplot['Variable'].isin(social_cap)]

input_dropdown_sc = alt.binding_select(options=['clustering_hs', 'volunteering_rate_hs'], name='Social Capital Factors')
selection_sc = alt.selection_single(fields=['Variable'], bind=input_dropdown_sc)

sc_scatter_chart = alt.Chart(social_cap_scatter, title="Social Capital Variables").mark_point(color="white").encode(
    x='Value:Q',
    y='bias_own_ses_hs:Q',
    tooltip='Variable:N'
).add_selection(
    selection_sc
).transform_filter(
    selection_sc
)

# Other demographic variables

other_demos = ['rent_pctage_household_income',
# 'num_white','num_black','num_native','num_asian', 'num_hispanic', 'num_ell',
'pct_commute_less_than_29','pct_commute_greater_than_30','pct_snap_no_cash_public_assist',
'pct_snap_with_cash_public_assist', 'pct_poverty_inc_ratio_185_under']

otherdems_scatter = densplot[densplot['Variable'].isin(other_demos)]

input_dropdown_od = alt.binding_select(options=['rent_pctage_household_income',
# 'num_white','num_black','num_native','num_asian', 'num_hispanic', 'num_ell',
'pct_commute_less_than_29','pct_commute_greater_than_30','pct_snap_no_cash_public_assist',
'pct_snap_with_cash_public_assist', 'pct_poverty_inc_ratio_185_under'], name='Census Block Demographic Factors')
selection_od = alt.selection_single(fields=['Variable'], bind=input_dropdown_od)

od_scatter_chart = alt.Chart(otherdems_scatter, title="Census Block Demographic Variables").mark_point(color="white").encode(
    x='Value:Q',
    y='bias_own_ses_hs:Q',
    tooltip='Variable:N'
).add_selection(
    selection_od
).transform_filter(
    selection_od
)


# Great Schools Variables
great_schools =[ 'ap_courses','honors_classes','gs_equity_rating',
'gs_overall_rating','gs_test_rating','gs_progress_rating']


# scatter
gs_scatter = densplot[densplot['Variable'].isin(great_schools)]

input_dropdown_gs = alt.binding_select(options=['ap_courses','honors_classes','gs_equity_rating',
'gs_overall_rating','gs_test_rating'], name='Great Schools Factors')
selection_gs = alt.selection_single(fields=['Variable'], bind=input_dropdown_gs)

gs_scatter_chart = alt.Chart(gs_scatter, title="Great Schools Variables").mark_point(color="white").encode(
    x='Value:Q',
    y='bias_own_ses_hs:Q',
    tooltip='Variable:N',
).add_selection(
    selection_gs
).transform_filter(
    selection_gs
)


#  variables charts
st.header('Identifying and exploring variables for predictive model')
st.markdown('We wanted to investigate a large swath of variables when predicting friending bias. Before building the models, we explored univariate relationships between these variables and friending bias on the school level. We found that most of the variables did not have a perfectly linear relationship with friending bias.')

col1, col2 = st.columns(2)

with col1:
    st.altair_chart(cps_scatter_chart)
    st.altair_chart(sc_scatter_chart)
with col2:
    st.altair_chart(od_scatter_chart)
    st.altair_chart(gs_scatter_chart)




# Testing models bar chart

st.header('Testing models')
st.markdown('To identify the best model to pursue, we tested 8 different machine learning regression types from the sklearn library. SVM and random forest yielded the highest R-squared values across a 5-fold cross validation. Even still, the predictive power of the top models hovered around <0.39, which is not substantive enough of an R-squared value to make the case for predicting friending bias. Therefore, when optimizing for friending bias in future rezoning projects, friending bias will be taken as given.')

reg_test_results = pd.read_csv("data/reg_model_test_results.csv")

test_results_bar = alt.Chart(reg_test_results, title="Models").mark_bar(color="grey").encode(
    # x='type:N',
    x=alt.X("type:N", sort=alt.EncodingSortField(field="avg_cv_score:q")),
     y='avg_cv_score:Q'
)
# idk why not sorting

error_bars = test_results_bar.mark_rule(color='white').encode(
    x='type:N',
    y='lower_CI:Q',
    y2= 'upper_CI:Q'

)

st.altair_chart(test_results_bar+error_bars, use_container_width=True)



st.header('Random forest feature importance')
st.markdown('We included over 25 variables in our training dataset that we believed may have a predictive effect on friending bias. Here, you can select any variables of interest to see their feature importance in the random forest model.')
rf_options = st.multiselect(
    'Variables',
    ['pct_homeowners',
  'pct_less_than_hs_edu',
  'pct_hs_degree',
  'pct_associates_degree',
  'pct_bachelors_degree',
  'pct_post_grad_higher_degree',
  'pct_single_parent_families',
  'pct_non_english_lang',
  'med_family_income',
  'clustering_hs',
  'volunteering_rate_hs',
  'rent_pctage_household_income',
  'num_white',
  'num_black',
  'num_native',
  'num_asian',
  'num_hispanic',
  'num_ell',
  'pct_commute_less_than_29',
  'pct_commute_greater_than_30',
  'pct_snap_no_cash_public_assist',
  'pct_snap_with_cash_public_assist',
  'pct_poverty_inc_ratio_185_under',
  'ap_courses',
  'honors_classes',
  'gs_equity_rating',
  'gs_overall_rating',
  'gs_test_rating'],
  ['clustering_hs',
  'med_family_income'])

st.markdown("Note that it takes a minimum of approximately 8 variables to get a positive R-squared value")
st.write('You selected:', rf_options)




# Progress bar 
progress_text = "Running random forest model with selected variables..."
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

# rf loop
rf_reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
fi_results = {"var":[],
           "feat_imp": []}

for options in rf_options:
  X1 = df_rf_sub.loc[:, options].values
  y1 = df_rf_sub.loc[:,'bias_own_ses_hs'].values
  X = np.array(X1).reshape(-1,1)
  y = np.array(y1).reshape(-1,1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
    # print(X_train, y_train)
  rf_reg.fit(X_train, y_train)
  cv_score= (cross_val_score(rf_reg, X_train, y_train, scoring='r2', cv=5)).mean()
  score = rf_reg.score(X_test, y_test)
  feat_imp_result = permutation_importance(rf_reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
  forest_importances = pd.Series(feat_imp_result.importances_mean, index=[options])
  # fi_results['subset'].append(nameof(subset))
  i = 0
  while i < len(forest_importances):
    fi_results['var'].append(forest_importances.index[i])
    fi_results['feat_imp'].append(forest_importances[i])
    # fi_results['cross_val_r2'].append(cv_score[i])
    i +=1
  fi_results_df = pd.DataFrame(data=fi_results)


print(fi_results_df)
print(cv_score)
print(score)

rf_bar_chart = alt.Chart(fi_results_df, title="Feature Importance").mark_bar().encode(
    # x='var:N',
    # y=alt.Y('feat_imp:Q', sort='y'),
    x='feat_imp:Q',
    y=alt.Y('var:N', sort='-x')
).configure_mark(
    color='white'
)

# st.metric(label="R-squared", value=score)
st.metric(label="R-squared (average R2 across 5-fold cross validation)", value=cv_score)
st.altair_chart(rf_bar_chart, use_container_width=True)