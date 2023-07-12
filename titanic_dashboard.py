import streamlit as st
import pandas as pd
import plotly.express as px
import os
from os.path import join
import plotly.graph_objects as go
import json
import numpy as np

st.set_page_config(
                   page_title = 'RMS TITANIC: 승객 국적에 따른 생존률 변화',
                   layout = "wide", 
                   initial_sidebar_state = "expanded",
                   page_icon=":ship:",

                   )

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# load data
path = join(os.path.abspath(""),"data/titanic_preprocessed.csv")
df = pd.read_csv(path)


# side bar
st.sidebar.header("Titanic `Dashboard`")


# head
st.markdown("# RMS TITANIC: 승객 국적에 따른 생존률 변화")
c1, c2, c3 = st.columns(3, gap = "large")
c1.metric("No. of Passengers on Board", "{0}".format(len(df)))
c2.metric("Total Survival Rate", "{0}%".format(round(df["survived"].value_counts(normalize = True)[1]*100,2)))
c3.metric("No. of Different Nationalities", "{0}".format(df["country"].nunique()) )
            

# Row1
headcount_df = pd.read_csv("./data/headcount_df.csv")
with open('./data/country_geo.json', 'r') as f:
    data = json.load(f)


row1_c1, row2_c2 = st.columns([4,2])

with row1_c1:
    @st.cache_data
    def plot_map():
        fig = px.choropleth_mapbox(headcount_df, 
                                   geojson=data, 
                                   locations="code", 
                                   color= "headcount",
                                   hover_name="country", 
                                   hover_data = {"code":False},
                                   color_continuous_scale="Blues",
                                   featureidkey = "properties.ISO_A3",
                                   mapbox_style="carto-positron",
                                   zoom= 1, 
                                   center = {"lat": 38.774033, "lon": -42.572260},
                                   opacity=0.7,
                                   labels={'headcount':'탑승 인원'}
                                  )
        fig.update_layout(
                          margin={"r":0,"t":0,"l":0,"b":0},
                          # legend=dict(title=None,
                          #             orientation = 'h',
                          #             y=1, yanchor="bottom",
                          #             x=0.5, xanchor="center"
                          #            )
                         )        
        st.plotly_chart(fig, theme= None, use_container_width=True)
    plot_map()    
    


with row2_c2:
    
    surv_rate = df.groupby(["country","survived"])["index"].count().unstack(fill_value = 0)
    surv_rate["total"] = (surv_rate["no"] + surv_rate["yes"])
    surv_rate["rate"] = surv_rate["yes"]/ (surv_rate["no"] + surv_rate["yes"])

    colors = [
              '#6BAED6',
              '#6BAED6',
              '#6BAED6',
              '#6BAED6',

              '#4292C6',
              '#4292C6',
              '#4292C6',

              '#2171B5',
              '#2171B5',

              '#08519C',
              '#08519C',
             ]
    
    g = surv_rate[surv_rate["total"] > 20].sort_values(by = "rate", ascending= True)
    fig = go.Figure(data = [go.Bar(x = g["rate"],
                                   y = g.index, 
                                   marker_color = colors,
                                   text = round(g["rate"],2),
                                   textposition = "auto",
                                   orientation='h',
                                  ),
                           ]
                   )
    fig.update_layout(autosize = False,
                      height = 450,
                      title_text = "국가별 탑승 승객 생존률",
                      title_x = 0.45,
                      title_y = 0.85,
                                  )
    fig.update_traces(width=0.8)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    

# Row2
# survival rate per country (with more than 20 passenger)

def min_max_scaling(x, series):
    return (x - min(series)) / (max(series) - min(series))

def preprocess_data(df):
    tmp_df = df.loc[(df["num_passenger"] >= 20) & (df["embarked"] != "B"), :].copy()
    tmp_df["survived"] = np.where(tmp_df["survived"] == "yes", 1, 0)
    emb_df = tmp_df.groupby(["country", "embarked"]).agg({"index":"count"}).unstack(fill_value = 0)
    rate_df = tmp_df.groupby(["country"]).agg({"fare":["mean","median","min"], "index":"count","survived":"mean"})
    merged_df = rate_df.merge(emb_df, left_on = rate_df.index, right_on = emb_df.index)
    del tmp_df, emb_df, rate_df
    merged_df.columns = ["country", "fare_mean","fare_median","fare_min","num_people","survival_rate","C","Q","S"]
    return merged_df

g = preprocess_data(df)   
g["scaled_fare"] = g["fare_median"].apply(lambda x : min_max_scaling(x, g["fare_median"]))
g = g.sort_values(by = "survival_rate")

st.markdown("**국가별별 집계 (탑승 인원 20명 이상)**")
_, sbox_c2= st.columns([5,5])
with sbox_c2:
    
    st.markdown("""
    <style>
    .description_1 {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="description_1">부유한 사업가 및 상류층에 속한 사람들은 일등석이나 이등석에 탑승하였고, 생존율이 상대적으로 더 높았습니다. \
    승객 생존율이 0.4 이상인 국가들의 경우 “Cherbourg”에서 승선한 사람들의 비율이 상대적으로 높은 것을 확인할 수 있습니다. \
    이는 많은 부유층 사람들이 “Cherbourg”에서 승선한 것으로 볼 수도 있지만, 레바논 승객들의 승선권 지불 금액은 낮은 편에 속합니다. \
    Cherbourg”에서 탑승한 승객들의 객실이 구명보트가 저장된 갑판에 더 가까운 위치에 있었다면 구명보트에 더 쉽게 접근할 수 있었을 것입니다. </p>'
    ,unsafe_allow_html=True)
    st.markdown("")
    
row2_c1, row2_c2 = st.columns((5,5))
with row2_c1:

    fig = go.Figure(data = [
                            go.Scatter(name = "지불한 승선권 요금의 중앙값(0 ~ 1)",x = g["country"], y = g["scaled_fare"], marker = {"color" : "#08519C"}),
                            go.Scatter(name = "생존률(0 ~ 1)",x = g["country"], y = g["survival_rate"], marker = {"color" : "#9C081E"}),
        
                           ]  
             )
    fig.update_layout(barmode='stack',
                      autosize = False,
                      # width = 500,
                      height = 600,
                      title_text = "승선권 요금에 따른 생존률",
                      title_x = 0.4,
                      title_y = 0.9,
                      yaxis_range=[0,1],
                     )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


with row2_c2:

    fig = go.Figure(data = [
                            go.Bar(name = "Cherbourg 항 승선 비율", x = g["country"], y = g["C"]/g["num_people"], marker = {"color" : "#08519C"}, width = 0.6),
                            go.Bar(name = "Queenstown 항 승선 비율", x = g["country"], y = g["Q"]/g["num_people"], marker = {"color" : "#4292C6"}, width = 0.6),
                            go.Bar(name = "Southampton 항 승선 비율", x = g["country"], y = g["S"]/g["num_people"], marker = {"color" : "#9ECAE1"}, width = 0.6),
                            go.Scatter(name = "생존률(0 ~ 1)",x = g["country"], y = g["survival_rate"], marker = {"color" : "#9C081E"}),
                            # go.Scatter(name = "승객들의 평균 지불 요금",x = g["country"], y = g["scaled_fare"], marker = {"color" : "#50B244"}),
                           ]
             )
    fig.update_layout(barmode='stack',
                      autosize = False,
                      # width = 500,
                      height = 600,
                      title_text = "승선 장소에 따른 생존률",
                      title_x = 0.4,
                      title_y = 0.9,
                     )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


# Row3
## class 별
def plot_g(value, column = None):
    g = df.loc[df[value_dict[selected_col]] == value,:]
    g = g.groupby(["region"])["survived"].value_counts().unstack(fill_value = 0)
    fig = go.Figure(data=[go.Bar(name='생존', 
                                 x=g.index, 
                                 y=g.iloc[:,1].values,
                                 marker = {"color" : "#08519C"},
                                ),
                          go.Bar(name='사망', 
                                 x=g.index, 
                                 y=g.iloc[:,0].values, 
                                 marker = {"color" : "#9C081E"},
                                ),
                                ])
    fig.update_layout(barmode='stack',
                      autosize = False,
                      # width = 500,
                      height = 550,
                      title_text = "지역별 {0} 승객 생존 여부".format(value),
                      title_x = 0.4,
                      title_y = 0.85,
                     )
    fig.update_traces(width=0.5)
    fig.update_xaxes(categoryorder='array', 
                     categoryarray= ["North America", "UK", "Northern Europe", "Southern Europe", "Western Europe", "Eastern Europe", "Middle East", "Others"],
                     tickangle=315)
    if column:
        column.plotly_chart(fig, theme="streamlit", use_container_width=True)
    else:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
def plot_p(column = None):
        region_vc = df["region"].value_counts()
        fig = go.Figure(data=[go.Pie(labels=region_vc.index, 
                                     values=region_vc, 
                                     hole = .3,
                                     marker_colors = px.colors.sequential.Blues_r)])
        fig.update_layout(autosize=False,
                          height=470,
                          title_text = "지역별 탑승 인원",
                          title_x = 0.27,
                         )
        if column:
            column.plotly_chart(fig, theme="streamlit", use_container_width=True)
        else:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)        

            
st.markdown("**지역 및 대륙별 집계**")

sbox_c1, sbox_c2, _, sbox_c3 = st.columns([1,1,3,5])
with sbox_c1:
    value_dict = {"좌석 등급":"class", "연령대":"age_category", "성별":"gender"}
    selected_col = st.selectbox("데이터 선택: ", 
                                value_dict.keys(),
                               )
with sbox_c2:
    if selected_col == "연령대":
        age_dict = {0 : "10대 미만",10: "10대",20: "20대",30:"30대",40: "40대",50: "50대",60: "60대",70: "70대"}
        col_value = age_dict[st.slider("", 0, 60, step = 10, label_visibility = "hidden")]
        
# with sbox_c3:
#     st.markdown("comment")

if selected_col == "좌석 등급":
    sbox_c3.markdown("""
    <style>
    .description_1 {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    sbox_c3.markdown('<p class="description_1">20세기 초는 미국에서 경제적인 성장과 번영의 시기였습니다. 이로 인해 부유한 계층이 형성되었으며, 이들은 타이타닉과 같은 명문호에서 일등석으로 여행하며 제공되는 편의 시설을 즐길 수 있는 재정적인 여건을 가지게 되었습니다. 따라서 일등석 생존자들의 국적이 영국이 아닌 미국이 압도적으로 높음을 확인할 수 있습니다. </p>'
    ,unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4, gap = "small")
    c1.text("")
    c1.text("")
    c1.text("")
    plot_p(c1)
    for i,column in enumerate([c2,c3,c4]):
        # column.subheader("지역별 {0}등석 승객 생존 여부".format(i+1))
        plot_g(sorted(df["class"].unique())[i], column)
elif selected_col == "성별":
    sbox_c3.markdown("""
    <style>
    .description_1 {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    sbox_c3.markdown('<p class="description_1">타이타닉의 대피 절차에서 "여성과 아이들 먼저" 원칙이 적용되었습니다. 이 원칙에 따라 여성과 어린이들이 먼저 구명보트에 탑승하는 것이 우선되었습니다. 따라서 모든 나라에 걸쳐 남성에 비해 여성 승객들의 생존율이 더 높음을 확인할 수 있습니다. </p>'
    ,unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.text("")
    c1.text("")
    c1.text("")
    plot_p(c1)
    for i, column in enumerate([c2,c3]):
        # column.subheader("지역별 {0} 승객 생존 여부".format(df["gender"].unique()[i]))
        plot_g(df["gender"].unique()[i], column)
else:
    sbox_c3.markdown("""
    <style>
    .description_1 {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    sbox_c3.markdown('<p class="description_1">20대 이하의 경우 영국과 북유럽 지역의 승객들이 많이 탑승했지만, 30대 이상부터는 미국과 영국 승객들의 비율이 상대적으로 높음을 확인할 수 있습니다. </p>'
    ,unsafe_allow_html=True)
    sbox_c3.markdown("")    
    c1, c2 = st.columns(2)
    c1.text("")
    c1.text("")
    c1.text("")
    plot_p(c1)
    plot_g(col_value, c2) 