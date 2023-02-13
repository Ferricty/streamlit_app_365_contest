import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np


st.set_page_config(page_title="365 Contest Dashboard", layout="wide")


# #Inject CSS with Markdown
hide_sts_stile = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """
st.markdown(hide_sts_stile,unsafe_allow_html=True)


@st.cache
def get_the_data():
    purchases_detailed = pd.read_csv("purchases_detailed.csv")
    courses_info = pd.read_csv("courses.info.csv")
    students_per_country = pd.read_csv("students_per_country1.csv",dtype={"country":str})
    student_detailed_register = pd.read_csv("student_detailed_register.csv")
    ratings_copy = pd.read_csv("prueba_recomendacion.csv")
    df_course_info = pd.read_csv("course_info_rec_detailed.csv")
    minutes_watched = pd.read_csv("minutes_watched.csv")
    quizzes = pd.read_csv("quizzes.csv")
    ex_country = pd.read_csv("ex_country_results.csv")
    return [purchases_detailed,courses_info,students_per_country,student_detailed_register,ratings_copy,df_course_info,minutes_watched,quizzes,ex_country]

purchases_detailed, courses_info, students_per_country,student_detailed_register, ratings_copy, df_course_info,minutes_watched,quizzes,ex_country = get_the_data()

col_registered,col_minutes_watched,col_percentage_answers,col_exam_average = st.columns(4)



#----- Expander ------
with st.expander("More detailed info ↓"):
    left_select,right_select = st.columns(2)
    value_month = None
    
    if st.button("Add all months"):
        value_month = purchases_detailed.Month.unique()
    with left_select:
        
        month = st.multiselect(
            "Select the Month:",
            options=purchases_detailed.Month.unique(),
            default=value_month,
        )
    with right_select:
        
        student_country = st.multiselect(
            "Select the Nationality:",
            options=purchases_detailed.student_country.unique(),
            default=None,
        ) 

with col_registered:
    #Registered
    value2_col_registered = 0
    reg_df = student_detailed_register.groupby(["student_country","date_registered_m_n","date_registered_m"]).sum().reset_index().sort_values("date_registered_m")
    if month == [] and student_country == []:
        value1_col_registered = reg_df["Cant"].sum()
        value2_col_registered=0
    elif student_country == [] and month != []:

        value1_col_registered = reg_df[(reg_df["date_registered_m_n"].isin(month))].sum()["Cant"]
        if len(month)>1:
            value2_col_registered = list(reg_df[(reg_df["date_registered_m_n"].isin(month))]["Cant"])[-1] - list(reg_df[(reg_df["date_registered_m_n"].isin(month))]["Cant"])[-2]
    elif month == [] and student_country != []:

        value1_col_registered=reg_df[(reg_df["student_country"].isin(student_country))].sum()["Cant"]
        if len(student_country)>1:
            value2_col_registered = list(reg_df[(reg_df["student_country"].isin(student_country))]["Cant"])[-1] - list(reg_df[(reg_df["student_country"].isin(student_country))]["Cant"])[-2]
    
    else:
        
        value1_col_registered=reg_df[(reg_df["student_country"].isin(student_country)) & (reg_df["date_registered_m_n"].isin(month))].sum()["Cant"]
        if len(month)>1 or len(student_country)>1:
            
            value2_col_registered=list(reg_df[(reg_df["student_country"].isin(student_country)) & (reg_df["date_registered_m_n"].isin(month))]["Cant"])[-1] - list(reg_df[(reg_df["student_country"].isin(student_country)) & (reg_df["date_registered_m_n"].isin(month))]["Cant"])[-2]
   
    st.metric(":blue[Registered Students]",f"{value1_col_registered}",f"{value2_col_registered}")
    
   
with col_minutes_watched:
    #Minutes watched
    value2_minutes_watched = 0
    
    if month == [] and student_country == []:
        value1_minutes_watched = minutes_watched["minutes_watched"].sum()
        value2_minutes_watched = 0
        
    elif student_country == [] and month != []:

        value1_minutes_watched = minutes_watched[(minutes_watched["date_watched_m_n"].isin(month))].groupby(["weekday","dayname"]).sum()["minutes_watched"].sort_index().sum()
        
        if len(month)>1:
            
            value2_minutes_watched = list(minutes_watched[(minutes_watched["date_watched_m_n"].isin(month))].groupby("date_watched_m_n")["minutes_watched"].sum())[-1] - list(minutes_watched[(minutes_watched["date_watched_m_n"].isin(month))].groupby("date_watched_m_n")["minutes_watched"].sum())[-2]
    
    elif month == [] and student_country != []:

        value1_minutes_watched = minutes_watched[(minutes_watched["student_country"].isin(student_country))].groupby(["weekday","dayname"]).sum()["minutes_watched"].sort_index().sum()
        
        if len(student_country)>1:
            value2_minutes_watched = list(minutes_watched[(minutes_watched["student_country"].isin(student_country))].groupby("date_watched_m_n")["minutes_watched"].sum())[-1] - list(minutes_watched[(minutes_watched["student_country"].isin(student_country))].groupby("date_watched_m_n")["minutes_watched"].sum())[-2]
    
    else:
        
        value1_minutes_watched = minutes_watched[(minutes_watched["date_watched_m_n"].isin(month))&(minutes_watched["student_country"].isin(student_country))].groupby(["weekday","dayname"]).sum()["minutes_watched"].sort_index().sum()
        
        if len(month)>1 or len(student_country)>1:
            
            value2_minutes_watched = list(minutes_watched[(minutes_watched["student_country"].isin(student_country))&(minutes_watched["date_watched_m_n"].isin(month))].groupby("date_watched_m_n")["minutes_watched"].sum())[-1] - list(reg_df[(reg_df["student_country"].isin(student_country)) & (reg_df["date_registered_m_n"].isin(month))]["Cant"])[-2]
   
    st.metric(":blue[Minutes watched]", f"{value1_minutes_watched:.1f}", f"{value2_minutes_watched:.1f} min")
        
with col_percentage_answers:
    
    correct = quizzes["percentage"][0]
    wrong = quizzes["percentage"][1]
    st.metric(":blue[Percentage of correct answers]", f"{correct:.2f} %", f"Wrong {wrong:.2f} %", delta_color="inverse")
  
    
    
with col_exam_average:
    # Exam results average
    value2_exam_average = 0
    
    if month == [] and student_country == []:
        value1_exam_average = ex_country["exam_result"].mean()
        value2_exam_average = 0
        
    elif student_country == [] and month != []:

        value1_exam_average = ex_country[(ex_country["date_exam_completed_m_n"].isin(month))]\
                                    .groupby(["date_exam_completed_m_n"])["exam_result"].mean()[0]
        
        if len(month)>1:
            
            value2_exam_average = ex_country[(ex_country["date_exam_completed_m_n"].isin(month))]\
                                    .groupby("date_exam_completed_m_n")["exam_result"].mean()[-1] - \
                                        ex_country[(ex_country["date_exam_completed_m_n"].isin(month))]\
                                        .groupby("date_exam_completed_m_n")["exam_result"].mean()[-2]
    
    elif month == [] and student_country != []:

        value1_exam_average = ex_country[(ex_country["student_country"].isin(student_country))]\
                                    .groupby(["student_country"])["exam_result"].mean()[0]
        
        if len(student_country)>1:
            value2_exam_average = ex_country[(ex_country["student_country"].isin(student_country))]\
                                        .groupby("student_country")["exam_result"].mean()[-1] - \
                                            ex_country[(ex_country["student_country"].isin(student_country))]\
                                            .groupby("student_country")["exam_result"].mean()[-2]
    
    else:
        
        value1_exam_average = ex_country[(ex_country["student_country"].isin(student_country)) & \
                                    (ex_country["date_exam_completed_m_n"].isin(month))]\
                                        .groupby(["student_country","date_exam_completed_m_n"])["exam_result"].mean()[0]
        
        if len(month)>1 or len(student_country)>1:
            
            value2_exam_average = ex_country[(ex_country["student_country"].isin(student_country)) & (ex_country["date_exam_completed_m_n"].isin(month))].groupby(["date_exam_completed_m_n","student_country"])["exam_result"].mean()[-1] - ex_country[(ex_country["student_country"].isin(student_country)) & (ex_country["date_exam_completed_m_n"].isin(month))].groupby(["date_exam_completed_m_n","student_country"])["exam_result"].mean()[-2]
   
    st.metric(":blue[Exams Score Average]", f"{value1_exam_average:.1f}", f"{value2_exam_average:.1f}")
    

def plot_line_dashboard(df,value):
    d = {0:"Month",
         1:"date"}
    fig = px.line(df,
                x=d[value],
                y="Cant",
                line_group="purchase_type",
                labels={"purchase_type":"Purchase Type",
                        "color":"","month":"Month",
                        "legend":"Legend","Cant":"Amount"},
                color="purchase_type",
                color_discrete_sequence=["steelblue","tomato","rebeccapurple"],
                title="Purchases January-October 2022",
                render_mode='svg',
                width=560,
                height=400)
    return fig


col1,col2 = st.columns(2,gap="large")
with col1:
    st.text("For a more detailed version of purchases please add each month")

    if month == [] and student_country == []:
       
        purchases_detailed = purchases_detailed.groupby(["Month_id","Month","purchase_type"]).sum().reset_index()
        minutes_watched = pd.DataFrame(minutes_watched.groupby(["weekday","dayname"])\
                                       .sum()["minutes_watched"].sort_index()).reset_index()
        
        figure1 = plot_line_dashboard(purchases_detailed,0)
        
    elif student_country == [] and month != []:
        student_detailed_register = student_detailed_register[(student_detailed_register.date_registered_m_n.isin(month))]
        purchases_detailed = purchases_detailed[(purchases_detailed.Month.isin(month))]
        minutes_watched = pd.DataFrame(minutes_watched[minutes_watched["date_watched_m_n"].isin(month)]
                                       .groupby(["weekday","dayname"])\
                                       .sum()["minutes_watched"]\
                                       .sort_index()).reset_index()
        
        figure1 = plot_line_dashboard(purchases_detailed,1)
        
    elif month == [] and student_country != []:
        purchases_detailed = purchases_detailed[(purchases_detailed.student_country.isin(student_country))]
        purchases_detailed = purchases_detailed.groupby(["Month_id","Month","purchase_type"]).sum().reset_index()
        minutes_watched = pd.DataFrame(minutes_watched[minutes_watched["student_country"].isin(student_country)]
                                       .groupby(["weekday","dayname"])\
                                       .sum()["minutes_watched"]\
                                       .sort_index()).reset_index()
        figure1 = plot_line_dashboard(purchases_detailed,0)
        
    else:
        purchases_detailed = purchases_detailed[(purchases_detailed.Month.isin(month)) & (purchases_detailed.student_country.isin(student_country))]
        minutes_watched = pd.DataFrame(minutes_watched[(minutes_watched["student_country"].isin(student_country))&(minutes_watched["date_watched_m_n"].isin(month))]
                                       .groupby(["weekday","dayname"])\
                                       .sum()["minutes_watched"]\
                                       .sort_index()).reset_index()
        figure1 = plot_line_dashboard(purchases_detailed,1)

    col1.plotly_chart(figure1,use_container_width=True)
    
    
    figure3 = px.pie(student_detailed_register,names="student_country",values="Cant",
                     labels={"student_country":"Country","legend":"Legend","Cant":"Amount"},
                     color="student_country",
                     color_discrete_sequence=["steelblue","tomato","rebeccapurple","darkturquoise","white","yellowgreen"],
                     title="Nationality of the registered students in January-October 2022",
                     width=500,height=500,
                     hole=0.6)
    
    figure3.update_traces(marker=dict(line=dict(color="black",width=2)))
    
    col1.plotly_chart(figure3,use_container_width=True)
    

with col2:
    st.info("The dataset info is from 2022 contest",icon="ℹ️")
    
    def highlight(x, color1,color2):
        return np.where(x == np.array(x.to_numpy()), f"color: {color1}; background-color: {color2};", None)
    
    

    courses_info = courses_info.rename({"course_title":"Course Titles","course_rating":"Course Rating"},axis=1)

    st.dataframe((courses_info.style.format(precision=2)).apply(highlight, color1='#ffffff',color2="#120011", 
                                                             subset=(slice(0,2,1),
                                                                     ["Course Titles",
                                                                      "#Evaluations",
                                                                      "Course Rating"])),
                                                                        height=250,width=650)
    
     # Recommendation System

    from sklearn.feature_extraction.text import TfidfVectorizer

  
    # Create object
    vect = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vect.fit_transform(df_course_info["clean_title"])


    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    import re

    def clean_titles(title):    
        return re.sub(r'[^a-zA-Z0-9 ]','',title)  

    df_course_info["clean_title"] = df_course_info["course_title"].apply(clean_titles)  
    
    
    def search_titles(titles):
        titles = clean_titles(titles)
        query_vec = vect.transform([titles])
        similarity = cosine_similarity(query_vec, tfidf).flatten()
        indices = np.argpartition(similarity, -5)[-5:]
        results = ratings_copy.iloc[indices].iloc[::-1]
        return results 
        
    def find_similar(course_id):
        similar_users =  ratings_copy[(ratings_copy["course_id"] == course_id) & (ratings_copy["course_rating"] > 4)]["student_id"].unique()
        similar_users_recs = ratings_copy[(ratings_copy["student_id"].isin(similar_users)) & (ratings_copy["course_rating"] > 4)]["course_id"]
        similar_users_recs = similar_users_recs.value_counts() / len(similar_users)
        similar_users_recs = similar_users_recs[similar_users_recs > .10]

        all_users = ratings_copy[(ratings_copy["course_id"].isin(similar_users_recs.index)) & (ratings_copy["course_rating"] > 4)]
        all_users_recs = all_users["course_id"].value_counts() / len(all_users["student_id"].unique())
        rec_percentages = pd.concat([similar_users_recs, all_users_recs], axis = 1)
        rec_percentages.columns = ["similar","all"]
    
        rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
        rec_percentages = rec_percentages.sort_values("score", ascending = False)
        return rec_percentages.head().merge(df_course_info, left_index = True, right_on = "course_id")[["score", "course_title"]].reset_index(drop=True)
    
    title = st.text_input("Try our Recomendation System based on the ratings from other users with similar preferences")
     
    if len(title) > 5:
        results = search_titles(title)
        course_id = results.iloc[0]["course_id"]
            
        st.text("\n".join(list(find_similar(course_id)["course_title"])))
        
    
    figure4 = px.bar(minutes_watched,x="dayname",
                     y="minutes_watched",
                     title = "<b>Relation between minutes watched and days of the week</b>",
                     labels={"dayname":"Days","minutes_watched":"Total"},hover_data="",
                     color_discrete_sequence=["#0550ff"]*len(minutes_watched),
                     width=500,height=500)
    figure4.update_layout(
        yaxis=(dict(showgrid=False))
    )
    
    col2.plotly_chart(figure4,use_container_width=True)


