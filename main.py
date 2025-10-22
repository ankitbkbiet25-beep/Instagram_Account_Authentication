import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
from model_resources.preprocessing import preprocess


train=pd.read_csv("Insta_train.csv")
test=pd.read_csv("Insta_test.csv")

dataset=pd.concat([train,test],ignore_index=True)
df=preprocess(dataset)



st.set_page_config(page_title="Instagram Dashboard",layout="centered")
st.title("Instagram Account Risk Dashboard")
st.subheader("Behavioral Insights for Account Authenticity and Predictions")

insights,prediction=st.tabs(["Data Insights",'Prediction'])

with insights:
    st.subheader("Data Insights")
    st.markdown("Key Metrics")
    
    def metric_card(label, value):
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(75,0,130,0.4) 0%, rgba(199,21,133,0.3) 50%, rgba(255,182,193,0.2) 100%);
            border: 1px solid rgba(255,182,193,0.3);
            border-radius: 18px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(255,255,255,0.05);
            min-width: 200px;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin: 10px 8px;
            color: #FDFDFD;
            font-weight: 600;
            text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.4);
        ">
            <div style="font-size:14px;">{label}</div>
            <div style="font-size:18px;">{value}</div>
        </div>
        """, unsafe_allow_html=True)


    kpi1,kpi2,kpi3=st.columns(3)
    with kpi1:
        metric_card("Total Accounts",len(df))
    with kpi2:
        profile_pic_ratio=df['profile pic'].mean()
        metric_card("Accounts with Profile Pic",f"{round(profile_pic_ratio*100,2)}%")
    with kpi3:
        external_url_ratio=df['external URL'].mean()
        metric_card("Accounts with External URL",f"{round(external_url_ratio*100)}%")
    kpi4,kpi5,kpi6=st.columns(3)
    with kpi4:
        name_equal=df['name==username'].mean()
        metric_card("Accounts with same Name-Username",f"{round(name_equal*100)}%")
    with kpi5:
        avg_suspicion_score=df['SuspicionScore'].mean()
        metric_card("Avg Suspicion Score",f"{round(avg_suspicion_score,2)}")
    with kpi6:
        high_suspicion_ratio=(df['SuspicionLevel']=='High').mean()
        metric_card("Percent High Suspicion",f"{round(high_suspicion_ratio*100,2)}")
    
    distribution,relation=st.tabs(["Dataset Distribution","Features Relation"])
    with distribution:
        st.markdown(f"**◉ Dataset shape : `{df.shape[0]}`rows ✕ `{df.shape[1]}`columns**")
        st.write("◉ Column types :")
        
        st.write(df.dtypes.value_counts())
        st.markdown("◉ Data Description :")
        st.table(df.describe().T)
        st.markdown("<h3 style='text-align: center;'> Numerical Features Visulaization</h3>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left;'> Distribution & Boxplots</h3>", unsafe_allow_html=True)
        
        df["fake"] = df["fake"].map({0: "Real", 1: "Fake"})
        
        numeric_col=['nums/length username', 'fullname words', 'nums/length fullname','description length', 'posts', 'followers', 'follows','follower_follow_ratio', 'follow_post_ratio', 'SuspicionScore']
        for col in numeric_col:
            col1,col2=st.columns(2)
            
            with col1:
                st.markdown(f"**◉ Distribution of {col}**")
                fig1,ax=plt.subplots(figsize=(5,3.5))
                sns.histplot(df[col],kde=True,bins=30,color='#FF6B6B',ax=ax)
                ax.axvline(df[col].mean(),color='green',linestyle='--',label='Mean')
                ax.axvline(df[col].median(),color='purple',linestyle='-.',label='Median')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig1)
                
            with col2:
                st.markdown(f"**◉ Boxplot of {col}**")
                fig2,ax2=plt.subplots(figsize=(5,3))
                sns.boxplot(x=df[col],ax=ax2,color="#90EE90")
                ax2.grid(True)
                st.pyplot(fig2)
                
        st.markdown("<h3 style='text-align: left;'>Categorical Features Visulaization</h3>", unsafe_allow_html=True)
        col1,col2,col3=st.columns(3)
        with col1:
            st.markdown("◉ Distrtibution of accounts with profile pic")
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.countplot(x=df['profile pic'],order=df["profile pic"].value_counts().index,ax=ax,palette='Pastel1')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            
        with col2:
            st.markdown("◉ Distrtibution of accounts with external URL")
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.countplot(x=df['external URL'],order=df["external URL"].value_counts().index,ax=ax,palette='Pastel1')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            
        with col3:
            st.markdown("◉ Distrtibution of same accounts Name")
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.countplot(x=df['name==username'],order=df["name==username"].value_counts().index,ax=ax,palette='Pastel1')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        
        col4,col5=st.columns(2)
        with col4:
            st.markdown("◉ Distrtibution of private accounts")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(x=df['private'],order=df["private"].value_counts().index,ax=ax,palette='Pastel1')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        
        with col5:
            st.markdown("◉ Distrtibution of SuspicionLevel")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(x=df['SuspicionLevel'],order=df["SuspicionLevel"].value_counts().index,ax=ax,palette='Pastel1')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
    
    with relation:
        corr=df[numeric_col].corr(method='pearson')
        st.markdown("<h4 style='text-align: center;'>Numerical Feature Correlation Heatmap</h4>", unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,6))
        sns.heatmap(corr,annot=True,cmap='Pastel1',fmt='.2f',linewidth=0.5,ax=ax)
        st.pyplot(fig)
        
        st.markdown("<h4 style='text-align: center;'>Categorical Features vs Fake</h4>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h6 style='text-align: center;'>◉ Account Profile pic Distribution by type</h6>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.countplot(x=df["profile pic"], hue=df["fake"], data=df, palette="Pastel1", ax=ax)
            ax.set_xlabel("Account Authenticity")
            ax.set_title("Profile Pic vs Fake")
            st.pyplot(fig)
        with col2:
            st.markdown("<h6 style='text-align: center;'>◉ External Url Account Distribution by type</h6>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.countplot(x=df["external URL"], hue=df["fake"], data=df, palette="Pastel1", ax=ax)
            ax.set_xlabel("Account Authenticity")
            ax.set_title("External URL vs Fake")
            st.pyplot(fig)
            
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<h6 style='text-align: center;'>◉ Same name Account Distribution by type</h6>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.countplot(x=df["name==username"], hue=df["fake"], data=df, palette="Pastel1", ax=ax)
            ax.set_xlabel("Account Authenticity")
            st.pyplot(fig)
        with col4:
            st.markdown("<h6 style='text-align: center;'>Private Account Distribution by type</h6>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.countplot(x=df["private"], hue=df["fake"], palette="Pastel1", ax=ax)
            ax.set_xlabel("Account Authenticity")
            st.pyplot(fig)
            
        col1,col2=st.columns(2)
        with col1:
            st.markdown("<h6 style='text-align: center;'>◉ Account Suspicion Level Distribution by type</h6>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.countplot(x=df["SuspicionLevel"], hue=df["fake"], palette="Pastel1", ax=ax)
            ax.set_xlabel("Account Authenticity")
            st.pyplot(fig)
            
        st.markdown("<h4 style='text-align: center;'>Numeric Features vs Fake</h4>", unsafe_allow_html=True)
        for feature in numeric_col:
            st.markdown(f"◉ {feature} vs Account Authenticity")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.violinplot(x=df["fake"], y=df[feature],data=df, palette="Pastel1", ax=ax)
                ax.set_xlabel("Account Authenticity")
                ax.grid(True)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.stripplot(x=df["fake"], y=df[feature],data=df,palette="Pastel1", jitter=True, alpha=0.9, ax=ax)
                ax.set_xlabel("Account Authenticity")
                ax.grid(True)
                st.pyplot(fig)
with prediction:
    model=joblib.load("model_resources/insta_voting_model.pkl")
    pt=joblib.load("model_resources/power_transformer.pkl")
    metrics_df=pd.read_csv("model_resources/model_metrics.csv")
    
    st.subheader("Account Authenticity Prediction")
    with st.form("Fill the Account Information"):
        profile_pic = st.radio("◉ Has Profile Picture", ["True", "False"]) == "True"
        username_length=st.number_input("◉ Username Numeric Ratio (numeric count / username length)",min_value=0.0,max_value=1.0)
        fullname_words=st.number_input("◉ Fullname Word Count",min_value=0)
        fullname_length = st.number_input("◉ Fullname Numeric Ratio (numeric characters/Fullname length)", min_value=0.0,max_value=1.0)
        name_equals_username = st.radio("◉ Name Equals Username", ["True", "False"]) == "True"
        description_length = st.number_input("◉ Description Length", min_value=0)
        external_url = st.radio("◉ Has External URL", ["True", "False"]) == "True"
        private = st.radio("◉ Is Private", ["True", "False"]) == "True"
        posts = st.number_input("◉ Number of Posts", min_value=0)
        followers = st.number_input("◉ Number of Followers", min_value=0)
        follows = st.number_input("◉ Number of Follows", min_value=0)
        
        submitted=st.form_submit_button("Predict")
    
    if(submitted):
        input_dict={
            'profile pic': profile_pic,
            'nums/length username': username_length,
            'fullname words': fullname_words,
            'nums/length fullname': fullname_length,
            'name==username': name_equals_username,
            'description length': description_length,
            'external URL': external_url,
            'private': private,
            'posts': posts,
            'followers': followers,
            'follows': follows
        }
        input_dict['follower_follow_ratio'] = followers/(follows + 1)
        input_dict['follow_post_ratio'] =follows/(posts + 1)
        input_dict['SuspicionScore'] = (
            int(not external_url) +
            int(not profile_pic) +
            int(name_equals_username) +
            int(posts == 0) +
            int(followers < 50) +
            int((input_dict['follower_follow_ratio'] < 0.1) and (followers > 1000)) +
            int((input_dict['follow_post_ratio'] > 20) and (posts < 10)) +
            int(description_length == 0)
    )
        score = input_dict['SuspicionScore']
        input_dict['SuspicionLevel'] = pd.cut([score], bins=[-1, 2, 5, 8], labels=['Low', 'Medium', 'High'])[0]

        
        numeric_col = ['nums/length username', 'fullname words', 'nums/length fullname','description length', 'posts', 'followers', 'follows','follower_follow_ratio', 'follow_post_ratio', 'SuspicionScore']
        categorical_cols = ['profile pic', 'external URL', 'name==username', 'private', 'SuspicionLevel']
        
        input_df=pd.DataFrame([input_dict])
        input_df[numeric_col]=pt.transform(input_df[numeric_col])
        input_df=pd.get_dummies(input_df,columns=categorical_cols,drop_first=True)
        
        expected_cols=model.feature_names_in_
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col]=0
                
        input_df=input_df[expected_cols]
        prediction=model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        label = "Fake" if prediction == 1 else "Real"
        if label == "Real":
            st.success(f"Prediction : **{label}**")
        else:
            st.warning(f"Prediction : **{label}**")
            
        st.write(f"Confidence: {max(prob):.2%}")
        
    st.header("**Model Performance**")
    st.dataframe(metrics_df.style.format({"Value": "{:.2%}"}),hide_index=True)
    