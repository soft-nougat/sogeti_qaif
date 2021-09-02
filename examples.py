# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 10:15:12 2021

@author: TNIKOLIC
"""
from __future__ import print_function
import streamlit as st
import streamlit.components.v1 as components
import helper as help
import shap 
## Imports libs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def tabular_bias():
    
    help.header("Tabular Data",
                is_sidebar = False)
    
    st.markdown("""---""")
    
    col1, col2, col3, col4 = st.beta_columns(4)
    
    with col1:
        help.sub_text("""<b>Keywords</b>""", 
                      alignment = "left")
        help.sub_text("""
                      Normalization
                      <br>Sampling 
                      <br>Stratisfied
                      <br>Bias
                      <br>Tabular data
                      <br>Imbalanced classes""", 
                      alignment = "left")
    with col2:
        help.sub_text("""<b>Packages used</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = https://pandas.pydata.org/>
                      Pandas</a>
                      <br><a href = https://scikit-learn.org/stable/>
                      SKLearn</a>""", 
                      alignment = "left")
    with col3:
        help.sub_text("""<b>Example dataset</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>
                      Titanic seaborn</a>""", 
                      alignment = "left")
    with col4:
        help.sub_text("""<b>Similar packages</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = ' https://github.com/fairlearn/fairlearn/tree/main/fairlearn/metrics'>
                      Fairlearn</a>""", 
                      alignment = "left")
        
    st.markdown("""---""")
        
    expander = st.beta_expander('Data Understanding', 
                                expanded=False)

    with expander:
        
        exp_text = """
        <b>The goal</b>
        <br>Getting the insight into the data, 
        <span style = "color:#F26531">
        <dfn title = "It could be that certain values of variables in your data have 
        strong predictive power, however, that this is unwanted or unethical for the 
        task at hand. E.g. an automated model for insurance approval could implicitily 
        learn that someone with an age over 30 has a higherchance of being accepted 
        than someone below the age of 30 with an exact similar situation.">
        detect potential vulnerable variables.</dfn></span></br>
        <b>Practical example</b>
        <br>In the upcoming example we load the commonly-known 
        <span style = "color:#F26531">
        <dfn title = "This data consists of information of all passengers that embarked the titanic. 
        Information such as: name, ticket, boat,age, sex, room and ticket-type are present.">
        titanic dataset.</dfn></span>
        The goal in this example is to predict the survival status of individual 
        passengers on the Titanic after training on the (sampled) dataset and 
        see what the effect of the sampling has on the model accuracy.</br>
        """
        
        help.sub_text(exp_text)
        
        button_du = st.button('Run Data Understanding Example')
        
        if button_du:
            # only execute this code when expanded + clicked
            with st.echo():
                import seaborn as sns
                titanic = sns.load_dataset('titanic')
                
                st.write("Number of males: "+ str(titanic['sex'].value_counts()['male']))
                st.write("Number of females: "+str(titanic['sex'].value_counts()['female']))
                
                sns.countplot(x='survived', data=titanic)
                st.pyplot()
                
             
            help.sub_text("""
                          From this we can observe that there were approximately twice as many males on board of the
                          Titanic compared to females. For exact proportions:
                          """)
                
            with st.echo():
                #Normalized
                st.write(titanic['sex'].value_counts(normalize=True))
                
                # If we now take a closer look at the data, it seems that sex is actually a relevant factor. From this
                # plot it appears that females seem to have a higher chance of survival compared to the males.
                
                # Countplot
                sns.catplot(x ="sex", hue ="survived", kind ="count", data = titanic)
                st.pyplot()
    
    
    expander = st.beta_expander('Data Preparation', 
                                expanded=False)

    with expander:
        
        exp_dp_text = """
        <b>The goal</b>
        <br>Prepare data for model training and ensure the training dataset is not biased.
        Ths can be achieved by the following methods:
        <li><span style = "color:#F26531">
        <dfn title = 
        "Stratified random sampling is one common method that is used by researchers because it
        enables them to obtain a sample population that best represents the entire population being
        studied, making sure that each subgroup of interest is represented.">
        - Stratified random sampling</span></dfn></li>
        <li><span style = "color:#F26531">
        <dfn title =
        "Oversampling can be used to avoid sampling bias in situations where members of defined
        groups are underrepresented (undercoverage). This is a method of selecting respondents from
        some groups so that they make up a larger share of a sample than they actually do the
        their population. After all data is collected, responses from oversampled groups are weighted to 
        actual share of the population to remove any sampling bias.">
        - Oversampling</span></dfn></li>
        <li><span style = "color:#F26531">
        <dfn title =
        "Sampling weights are intended to compensate for the selection of specific observations 
        with unequal probabilities (oversampling), non-coverage, non-responses, and other types of 
        bias. If a biased data set is not adjusted, population descriptors (e.g., mean, median) 
        will be skewed and fail to correctly represent the population’s proportion to the population.">
        - Adjusting weights</span></dfn></li>
        <li><span style = "color:#F26531">
        <dfn title = "Define a target population and a sampling frame (the list of individuals that 
        the sample will be drawn from). Match the sampling frame to the target population 
        as much as possible to reduce the risk of sampling bias.">
        - Estimate missing data of classes</span></dfn></li>
        <b>Practical example</b>
        <br>First we have to do some <span style = "color:#F26531">
        <dfn title = "This entails filling in the missing values, removing noninformative
        variables and basic one-hot-encoding for categorical variables.">
        standard preprocessing </span></dfn> of the data so we can work with it and it is
        ready to be inserted into a model. For now we just import the preprocessed dataset with 
        similar sex-ratio's as previously explored.
        """
        
        help.sub_text(exp_dp_text)
        
        button_dp = st.button('Run Data Preparation Example')
        
        if button_dp:
            # only execute this code when expanded + clicked
            with st.echo():
                
                import pandas as pd
                from sklearn.model_selection import train_test_split
                
                data = pd.read_csv("examples_data/titanic_cleaned.csv")
                
                ############ "Random sampling" - split the dataset in train and test - force low amount of women in trainset
                # Note; this is not actually random, but could appear worst-case, just for illustration
                
                del data['Unnamed: 0']
                sorted_data = data.sort_values(["Sex_female"], ascending=True)
                X, y = sorted_data.iloc[:, 1:], data.iloc[:, 0]
                
                # Take first 300 males for traindata + 3 females
                X_train_rnd = X[:200]
                y_train_rnd = y[:200]
                X_train_rnd = X_train_rnd.append(X.iloc[-25:-22])
                y_train_rnd = y_train_rnd.append(y.iloc[-25:-22])
                
                # Take 20 females as testdata
                X_test_female = X.tail(20)
                y_test_female = y.tail(20)
                
                # Take 20 males to compare for testing
                X_test_rnd_male = X.iloc[210:230]
                y_test_rnd_male = y.iloc[210:230]
                
                ############ Stratisfied sampling - keep proportions of original data
                # Let's keep the 20 women test-set and deduct them
                # Use the rest of the data to train and test with stratified split
                X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split( X[0:-20], y[0:-20], test_size=0.33, random_state=41, stratify=X[0:-20]['Sex_female'])
                
                # Check proportions after random sampling
                st.write("Proportion of females in trainset after stratisfied sampling: "+ str())
                st.write(len(X_train_strat[X_train_strat['Sex_female']==1])/len(X_train_strat))
                
                

    expander = st.beta_expander('Model Development', 
                                expanded=False)

    with expander:
        
        exp_dp_text = """
        <b>The goal</b>
        <br>Given that we have handled the data in previous 2 phases, here we need to 
        detect bias neglection during model development.
        <br><b>Practical example</b>
        <br>For the sake of this example we show the difference in model performance between 
        a model trained on an imbalanced compared to a balanced dataset.
        """
        
        help.sub_text(exp_dp_text)
        
        button_md = st.button('Run Model Development Example')
        
        if button_md:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            
            data = pd.read_csv("examples_data/titanic_cleaned.csv")
            
            ############ "Random sampling" - split the dataset in train and test - force low amount of women in trainset
            # Note; this is not actually random, but could appear worst-case, just for illustration
            
            del data['Unnamed: 0']
            sorted_data = data.sort_values(["Sex_female"], ascending=True)
            X, y = sorted_data.iloc[:, 1:], data.iloc[:, 0]
            
            # Take first 300 males for traindata + 3 females
            X_train_rnd = X[:200]
            y_train_rnd = y[:200]
            X_train_rnd = X_train_rnd.append(X.iloc[-25:-22])
            y_train_rnd = y_train_rnd.append(y.iloc[-25:-22])
            
             # Take 20 females as testdata
            X_test_female = X.tail(20)
            y_test_female = y.tail(20)
            
            # Take 20 males to compare for testing
            X_test_rnd_male = X.iloc[210:230]
            y_test_rnd_male = y.iloc[210:230]
            X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split( X[0:-20], y[0:-20], test_size=0.33, random_state=41, stratify=X[0:-20]['Sex_female'])
            
            # only execute this code when expanded + clicked
            with st.echo():
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.metrics import confusion_matrix
                
                def accuracy_score(y_pred, y_test):
                    a = list(y_pred)
                    b = list(y_test)
                    accuracy = len([a[i] for i in range(0, len(a)) if a[i] == b[i]]) / len(a)
                    return accuracy
                
                clf = DecisionTreeClassifier()
                clf = clf.fit(X_train_rnd,y_train_rnd)
                
                # Show accuracy on men
                y_pred = clf.predict(X_test_rnd_male)
                st.write("Accuracy of the model on men: ", str(accuracy_score(y_test_rnd_male, y_pred)))

                # Show accuracy on women
                y_pred = clf.predict(X_test_female)
                st.write("Accuracy of the model on the 20 women: "+ str(accuracy_score(y_test_female, y_pred))+"\n")
                # In depth look of errors
                tn, fp, fn, tp = confusion_matrix(y_test_female, y_pred).ravel()
                st.write("Number of False positives of 20 women: "+ str(fp))
                st.write("Number of False negatives of 20 women: "+ str(fn))
                
                #show why better generalization, specifically on women with stratisfied example
                clf = DecisionTreeClassifier()
                clf = clf.fit(X_train_strat, y_train_strat)
                
                # Show accuracy on women
                y_pred = clf.predict(X_test_female)
                st.write("Accuracy of the model on the 20 women: "+ str(accuracy_score(y_test_female, y_pred))+"\n")
                
def text_bias():
    
    help.header("Text Data",
                is_sidebar = False)
    
    st.markdown("""---""")
    
    col1, col2, col3, col4 = st.beta_columns(4)
    
    with col1:
        help.sub_text("""<b>Keywords</b>""", 
                      alignment = "left")
        help.sub_text("""
                      Normalization
                      <br>Sampling 
                      <br>Stratisfied
                      <br>Bias
                      <br>Text data
                      <br>Imbalanced classes""", 
                      alignment = "left")
    with col2:
        help.sub_text("""<b>Packages used</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = "https://github.com/ResponsiblyAI/responsibly">
                      Responsibly</a> 
                      """, 
                      alignment = "left")
    with col3:
        help.sub_text("""<b>Example dataset</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = "https://data.world/jaredfern/google-news-200d">
                      Google News</a>""", 
                      alignment = "left")
    with col4:
        help.sub_text("""<b>Similar packages</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = "https://github.com/dccuchile/wefe">
                      WEFE</a>""", 
                      alignment = "left")
        
    st.markdown("""---""")
    
    expander = st.beta_expander('Data Understading', 
                                expanded=False)

    with expander:
        
        exp_text = """
        When usng text data in AI applications, we run into a problem of feeding the semantics and 
        meaning of text to the model as training data. In order to achieve this, 
        <span style = "color:#F26531">
        <dfn title = "Dense vector representations of words trained from document corpora.">
        word embeddings</dfn></span> are used.
        Even though they are a crucial part of NLP pipelines, they have a negative side of
        learning bias from the text corpora they are trained on.
        """
        
        help.sub_text(exp_text)
        
    expander = st.beta_expander('Responsibly Example', 
                                expanded=False)

    with expander:
        
        exp_text = """
        <b>The method</b>
        <br>We will be using  
        <span style = "color:#F26531">
        <dfn title = 
        "Metrics and debiasing methods for bias (such as gender and race) in word embedding.">
        Responsibly WE</dfn></span> to score the gender bias for certain 
        <span style = "color:#F26531">
        <dfn title = 
        "A vector of words. The attribute words describe professions by which 
        a bias towards one of the gender groups may be exhibited (e.g., doctor, nurse).">
        attributes</dfn></span> and 
         <span style = "color:#F26531">
        <dfn title = 
        "A vector of words. The target describe the gender groups in which 
        fairness is intended to be measured (e.g., women, men, non-binary).">
        targets</dfn></span>.
        <br>Responsibly WE implements the following metrics:
        <li>Word Embedding Association Test (WEAT)
        <li>Bias measure and debiasing
        <li>Clustering as classification of biased neutral words</li>
        <br>In the following code, we measure the direct gender bias of a pretrained 
        Responsibly WE model, we debias it and run correlations to validate the debiasing.
        """
        
        help.sub_text(exp_text)
        
        button_res = st.button('Run Responsibly Example')
        
        if button_res:
            with st.echo():
                
               from responsibly.we import load_w2v_small
               
               # load a pretrained model on google news data
               w2v_small = load_w2v_small()
                
               # get the most similar terms from model, 
               # exclude words related to he/him
               she = w2v_small.most_similar(positive=['doctor', 'she'],
                                               negative=['he'])
               
               
               # get the most similar terms from model, 
               # exclude words related to she/her
               he = w2v_small.most_similar(positive=['doctor', 'he'],
                                                    negative=['she'])
               
               # visualise data
               import matplotlib.pyplot as plt; plt.rcdefaults()
               import numpy as np
               import matplotlib.pyplot as plt
                
               names = list(k[0] for k in she)
               values = list(k[1] for k in she)
                
               y_pos = np.arange(len(she))
                
               plt.figure(figsize=(20,10))
               plt.bar(y_pos, values, align='center', alpha=0.5)
               plt.xticks(y_pos, names)
               plt.ylabel('Correlation')
               plt.title('Profession/term')
                
               st.pyplot()
               
               names = list(k[0] for k in he)
               values = list(k[1] for k in he)
                
               y_pos = np.arange(len(he))
                
               plt.figure(figsize=(20,10))
               plt.bar(y_pos, values, align='center', alpha=0.5)
               plt.xticks(y_pos, names)
               plt.ylabel('Correlation')
               plt.title('Profession/term')
                
               st.pyplot()
               
               # get bias score and debias model
               from responsibly.we import GenderBiasWE
               
               direct_bias = GenderBiasWE(w2v_small).calc_direct_bias()
               st.write("The direct bias score is: ", direct_bias)
               GenderBiasWE(w2v_small).debias()
               unbiased = GenderBiasWE(w2v_small).calc_direct_bias()
               st.write("The direct bias score after debiasing is: ", unbiased)
               
               # run example again
               # get the most similar terms from model, 
               # exclude words related to he/him
               she = w2v_small.most_similar(positive=['doctor', 'she'],
                                               negative=['he'])
               
               
               # get the most similar terms from model, 
               # exclude words related to she/her
               he = w2v_small.most_similar(positive=['doctor', 'he'],
                                                    negative=['she'])
               
               names = list(k[0] for k in she)
               values = list(k[1] for k in she)
                
               y_pos = np.arange(len(she))
                
               plt.figure(figsize=(20,10))
               plt.bar(y_pos, values, align='center', alpha=0.5)
               plt.xticks(y_pos, names)
               plt.ylabel('Correlation')
               plt.title('Profession/term')
                
               st.pyplot()
               
               names = list(k[0] for k in he)
               values = list(k[1] for k in he)
                
               y_pos = np.arange(len(he))
                
               plt.figure(figsize=(20,10))
               plt.bar(y_pos, values, align='center', alpha=0.5)
               plt.xticks(y_pos, names)
               plt.ylabel('Correlation')
               plt.title('Profession/term')
                
               st.pyplot()
               
        
def tabular_xai():
    
    help.header("Tabular Data",
                is_sidebar = False)
    
    st.markdown("""---""")
    
    col1, col2, col3, col4 = st.beta_columns(4)
    
    with col1:
        help.sub_text("""<b>Keywords</b>""", 
                      alignment = "left")
        help.sub_text("""
                      Explainability
                      <br>Visualisation 
                      <br>Transparency""", 
                      alignment = "left")
    with col2:
        help.sub_text("""<b>Packages used</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = 'https://github.com/slundberg/shap'>Shap</a>
                      """, 
                      alignment = "left")
    with col3:
        help.sub_text("""<b>Example dataset</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29>
                      Breast Cancer Dataset</a>""", 
                      alignment = "left")
    with col4:
        help.sub_text("""<b>Similar packages</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = 'https://github.com/marcotcr/lime'>Lime</a>""", 
                      alignment = "left")
        
    st.markdown("""---""")
    
    expander = st.beta_expander('Data Understanding', 
                                expanded=False)

    with expander:
        
        exp_text = """
        In the example dataset, features are computed from a digitized image of a 
        Fine Needle Aspirate (FNA) of a breast mass. They describe characteristics 
        of the cell nuclei present in the image. The task is a binary classification. 
        """
        
        help.sub_text(exp_text)
    
    
    
    expander = st.beta_expander('Model Development and Evaluation', 
                                expanded=False)
    
    with expander:
        
        exp_text = """
       In this example, we take a binary classification case built with a sklearn model. 
       We train, tune and test our model. Then we can use our data and the model to create 
       an additional <span style = "color:#F26531">
<dfn title = 
"A XAI method based on a game theory approach to explain individual predictions. 
The feature values of a data instance act as players in coalition. The Shapley 
value is the average marginal contribution of a feature value across all possible 
coalitions."> 
       SHAP model </dfn></span> that explains our classification model – a breast cancer 
       prediction outcome.  
       """
        
        help.sub_text(exp_text)
        
        button_me = st.button('Run Example')
        
        if button_me:
            with st.echo():
                import pandas as pd
                import xgboost as xgb
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                # import the dataset from Sklearn
                from sklearn.datasets import load_breast_cancer
                
                
                # Read the DataFrame, first using the feature data
                data = load_breast_cancer() 
                df = pd.DataFrame(data.data, columns=data.feature_names)
                
                
                # Add a target column, and fill it with the target data
                df['target'] = data.target
                
                df.head()
                
                # Set up the data for modelling 
                y = df['target'].to_frame() # define Y
                X = df[df.columns.difference(['target'])] # define X
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # create train and test
            
            	# build model - Xgboost
                xgb_mod = xgb.XGBClassifier(random_state=42,gpu_id=0) # build classifier
                xgb_mod = xgb_mod.fit(X_train,y_train.values.ravel()) 
            	
            
            	# make prediction and check model accuracy 
                y_pred = xgb_mod.predict(X_test)
            	
                accuracy = accuracy_score(y_test, y_pred)
                st.write("Accuracy: %.2f%%" % (accuracy * 100.0))
                
                # EVALUATION
                import shap
                # Generate the Tree explainer and SHAP values
                explainer = shap.TreeExplainer(xgb_mod)
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value
                
                ############## visualizations #############
                # Generate summary dot plot
                shap.summary_plot(shap_values, X,title="SHAP summary plot") 
                st.pyplot()
                st.write("<b>SHAP summary dot plot</b>:  Depicts how different feature values ",
                         "contribute globally. ",
                         "Shows feature importances and their impact on the model.",
                         unsafe_allow_html=True)
                # Generate summary bar plot 
                shap.summary_plot(shap_values, X,plot_type="bar") 
                st.pyplot()
                st.write("<b>SHAP Bar plot</b>: Global explanations depicting feature importance.",
                        unsafe_allow_html=True)
                # Generate waterfall plot  
                shap.plots._waterfall.waterfall_legacy(expected_value, 
                                                       shap_values[79], 
                                                       features=X.loc[79,:], 
                                                       feature_names=X.columns,
                                                       max_display=15, show=True)
                st.pyplot()
                st.write("<b>SHAP waterfall plot</b>: The waterfall plot also allows us to see  ",
                         "the amplitude and the nature of the impact of a feature with its ",
                         "quantification. It also allows to see the order of importance of the ",
                         "features and the values taken by each feature for the studied sample.",
                        unsafe_allow_html=True)
                # Generate dependence plot
                shap.dependence_plot("worst concave points", shap_values, X, 
                                     interaction_index="mean concave points")
                st.pyplot()
                # Generate multiple dependence plots
                for name in X_train.columns:
                     shap.dependence_plot(name, shap_values, X)
                shap.dependence_plot("worst concave points", shap_values, X, 
                                     interaction_index="mean concave points")
                st.pyplot()
                st.write("<b>SHAP dependence plot</b>: analyzes the features two by two by ",
                        "suggesting a possibility to observe the interactions.",
                       unsafe_allow_html=True)
                # Generate force plot - Multiple rows
                # use st_shap function
                st_shap(shap.force_plot(explainer.expected_value, shap_values[:100,:], 
                                        X.iloc[:100,:]), 400)
                # Generate force plot - Single
                st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], 
                                        X.iloc[0,:]))
                st.write("<b>SHAP force plot</b>: The force plot is good to see where the ",
                         "“output value” fits in relation to the “base value”. We also ",
                         "see which features have a positive (red) or negative (blue) ",
                         "impact on the prediction and the magnitude of this impact.",
                      unsafe_allow_html=True)
                # Generate Decision plot 
                shap.decision_plot(expected_value, shap_values[79],link='logit' ,features=X.loc[79,:], feature_names=(X.columns.tolist()),show=True,title="Decision Plot")
                st.pyplot()
                st.write("<b>SHAP decision plot</b>: the decision plot makes it possible ",
                         "to observe the amplitude of each change, “a trajectory” taken by",
                         " a sample for the values of the displayed features.",
                      unsafe_allow_html=True)              


def image_xai():
    
    help.header("Image Data",
                is_sidebar = False)
    
    st.markdown("""---""")
    
    col1, col2, col3, col4 = st.beta_columns(4)
    
    with col1:
        help.sub_text("""<b>Keywords</b>""", 
                      alignment = "left")
        help.sub_text("""
                      Explainability
                      <br>Visualisation 
                      <br>Transparency""", 
                      alignment = "left")
    with col2:
        help.sub_text("""<b>Packages used</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = "https://github.com/alvinwan/neural-backed-decision-trees">
                      NBDT</a>
                      <br><a href = 'https://github.com/slundberg/shap'>Shap</a></br>
                      """, 
                      alignment = "left")
    with col3:
        help.sub_text("""<b>Example dataset</b>""", 
                      alignment = "left")
        help.sub_text("""
                      ResNet18 
                      <br>WideResNet28x10""", 
                      alignment = "left")
    with col4:
        help.sub_text("""<b>Similar packages</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = 'https://github.com/marcotcr/lime'>Lime</a>""", 
                      alignment = "left")
        
    st.markdown("""---""")
    
    expander = st.beta_expander('NBDT Example', 
                                expanded=False)

    with expander:
        
        exp_text = """
        For addressing the challenge of transparency in image classification use cases, 
        we can understand the model’s sequential decisions by creating hierarchical explanations. 
        We do this by using a new state of the art method, 
        <span style = "color:#F26531">
<dfn title =
"NBDTs aim to provide intermediate model decisions, not only the final prediction 
so that the user has visibility into the model. The approach is to convert a neural 
network into a decision tree by training it with a tree supervision loss. A hierarchy
is then generated in order to get the hierarchical or sequential explanations.
With this approach, the accuracy of a neural net is preserved.">
        Neural-Backed Decision Trees (NBDTs).</dfn></span> A standalone Streamlit
        App where you can try this out can be accessed 
        <a href = "https://share.streamlit.io/marktensensgt/streamlit_object_recognition/main.py">
        here</a>.
        <br>A demo is below. 
        """
        
        help.sub_text(exp_text)
        
        st.video("https://youtu.be/cSeAhiZB8SI")
        
    
    expander = st.beta_expander('Lime Example', 
                                expanded=False)
    
    with expander:
        
        exp_text = """
        In this example, we will use read an image and use the pre-trained InceptionV3 
        model available in Keras to predict the class of each image and then generate 
        explanation using a  <span style = "color:#F26531">
        <dfn title = "Lime stands for Local Interpretable Model-agnostic Explanations. 
It is a technique that explains how the input features of a machine learning model affect
its predictions. For example, in image classification tasks, LIME finds the region of 
an image (set of super-pixels) with the strongest association with a prediction label.  
">LIME model.</dfn></span> LIME generates explanations by creating a new dataset of 
random perturbations (with their respective predictions) around the instance being 
explained and then fitting a weighted local surrogate model. This local model is 
usually a simpler model with intrinsic interpretability such as a linear regression model. 
Once a linear model is fitted, we get a coefficient for each super-pixel in the 
image that represents how strong is the effect of the super-pixel in the prediction
 of Labrador.  
       """
        
        help.sub_text(exp_text)
        
        button_xai_me = st.button('Run Example')
        
        if button_xai_me:
            with st.echo():
                
                import numpy as np
                import skimage.transform
                import skimage.io
                Xi = skimage.io.imread("https://arteagac.github.io/blog/lime_image/img/cat-and-dog.jpg")
                Xi = skimage.transform.resize(Xi, (299,299)) 
                Xi = (Xi - 0.5)*2 #Inception pre-processing
                skimage.io.imshow(Xi/2+0.5) # Show image before inception preprocessing
                
                #Predict class for image using InceptionV3
                import keras
                from keras.applications.imagenet_utils import decode_predictions
                from keras.applications import inception_v3
                inceptionV3_model = keras.applications.inception_v3.InceptionV3() #Load pretrained model
                preds = inceptionV3_model.predict(Xi[np.newaxis,:,:,:])
                top_pred_classes = preds[0].argsort()[-5:][::-1] # Save ids of top 5 classes
                decode_predictions(preds)[0] #Print top 5 classes
                
                #Generate segmentation for image
                import skimage.segmentation
                superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
                num_superpixels = np.unique(superpixels).shape[0]
                skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))
                
                #Generate perturbations
                num_perturb = 150
                perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
                
                #Create function to apply perturbations to images
                import copy
                def perturb_image(img,perturbation,segments): 
                  active_pixels = np.where(perturbation == 1)[0]
                  mask = np.zeros(segments.shape)
                  for active in active_pixels:
                    mask[segments == active] = 1 
                  perturbed_image = copy.deepcopy(img)
                  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
                  return perturbed_image
                
                predictions = []
                for pert in perturbations:
                  perturbed_img = perturb_image(Xi,pert,superpixels)
                  pred = inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:])
                  predictions.append(pred)
                
                predictions = np.array(predictions)
                print(predictions.shape)
                
                #Show example of perturbations
                print(perturbations[0]) 
                skimage.io.imshow(perturb_image(Xi/2+0.5,perturbations[0],superpixels))
                
                #Compute distances to original image
                import sklearn.metrics
                original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
                distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
                print(distances.shape)
                
                #Transform distances to a value between 0 an 1 (weights) using a kernel function
                kernel_width = 0.25
                weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
                print(weights.shape)
                
                #Estimate linear model
                from sklearn.linear_model import LinearRegression
                class_to_explain = top_pred_classes[0] #Labrador class
                simpler_model = LinearRegression()
                simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
                coeff = simpler_model.coef_[0]
                
                #Use coefficients from linear model to extract top features
                num_top_features = 4
                top_features = np.argsort(coeff)[-num_top_features:] 
                
                #Show only the superpixels corresponding to the top features
                mask = np.zeros(num_superpixels) 
                mask[top_features]= True #Activate top superpixels
                st.image(skimage.io.imshow(perturb_image(Xi/2+0.5,mask,superpixels)))
                
                
def text_xai():
    
    help.header("Text Data",
                is_sidebar = False)
    
    st.markdown("""---""")
    
    col1, col2, col3, col4 = st.beta_columns(4)
    
    with col1:
        help.sub_text("""<b>Keywords</b>""", 
                      alignment = "left")
        help.sub_text("""
                      Explainability
                      <br>Visualisation 
                      <br>Transparency""", 
                      alignment = "left")
    with col2:
        help.sub_text("""<b>Packages used</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = 'https://github.com/marcotcr/lime'>Lime</a> 
                      """, 
                      alignment = "left")
    with col3:
        help.sub_text("""<b>Example dataset</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = "https://www.kaggle.com/c/nlp-getting-started/data">
                      Kaggle’s Disaster Tweets NLP Challenge</a>""", 
                      alignment = "left")
    with col4:
        help.sub_text("""<b>Similar packages</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = 'https://github.com/slundberg/shap'>Shap</a>""", 
                      alignment = "left")
        
    st.markdown("""---""")
    
    expander = st.beta_expander('LIME Explanation Example', 
                                expanded=False)

    with expander:
        
        exp_text = """
        <b>Introduction</b>
        <br>The implementation in this example is a post-hoc, model-agnostic interpretability 
        for both global and local model behaviour using 
        <span style = "color:#F26531">
<dfn title =
"LIME is model-agnostic and provides local model interpretability which 
means that it modifies a single data sample by tweaking the feature values
 and observes the resulting impact on the output.
The output of LIME is a list of explanations, reflecting the contribution 
of each feature to the prediction of a data sample. 
This provides local interpretability, and it also allows to determine 
which feature changes will have most impact on the prediction. 
">LIME</dfn></span>. 
        Using the Kaggle’s Disaster Tweets NLP Challenge; we will interpret which words 
        contributed to the probability of a logistic regression model predicting a tweet 
        referring to a real disaster or not. The model will be trained solely on the text
        body of the tweet. 
        <br><b>How a LIME explanation is created</b>: 
        <li>1. First, you choose the single prediction which you would like explained
        <li>2. LIME creates permutations of your data at this instance and collects 
        the black-box model results
        <li>3. It then gives weights to the new samples based on how closely they 
        match the data of the original prediction
        <li>4. A new, less complex, interpretable model is trained on the data variations
        created using the weights attached to each variation
        <li>5. Finally, the prediction can be explained by this local interpretable model
        """
        
        help.sub_text(exp_text)
        
        button_xai_txt = st.button('Run Example')
        
        if button_xai_txt:
            with st.echo():
                
                #importing libraries
                import pandas as pd
                import string
                import re
                import nltk
                import numpy as np
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn import metrics
                from sklearn.pipeline import make_pipeline
                from lime.lime_text import LimeTextExplainer
            
            	# downloading the nltk data for preprocessing
                nltk.download('stopwords')
                nltk.download('punkt')
            	
            
            	# downloading the data
                data_urls = ['https://raw.githubusercontent.com/KaliaBarkai/KaggleDisasterTweets/master/Data/%s.csv'%ds for ds in ['train', 'test', 'sample_submission']]
            	
            
            	# reading the data as pandas dataframe
                train = pd.read_csv(data_urls[0])
            
            	########### DATA PREPARATION #############
            	# remove urls, handles, and the hashtag from hashtags 
            	# (taken from https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression)
                def remove_urls(text):
                    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
                    return new_text
            	
            
            	# make all text lowercase
                def text_lowercase(text): 
                    return text.lower()
            	
            
            	# remove numbers
                def remove_numbers(text): 
                    result = re.sub(r'\d+', '', text) 
                    return result
            	
            
            	# remove punctuation
                def remove_punctuation(text): 
                    translator = str.maketrans('', '', string.punctuation)
                    return text.translate(translator)
            	
            
            	# function for all pre-processing steps
                def preprocessing(text):
                    text = text_lowercase(text)
                    text = remove_urls(text)
                    text = remove_numbers(text)
                    text = remove_punctuation(text)
                    return text
            	
            
            	# pre-processing the text body column
                pp_text = []
                for text_data in train['text']:
            	  # check if string
                  if isinstance(text_data, str):
                      pp_text_data = preprocessing(text_data)
                      pp_text.append(pp_text_data)
            	   # if not string
                  else:
                       pp_text.append(np.NaN)
            	
            
            # add pre-processed column to dataset
            train['pp_text'] = pp_text
            
            ################# DEVELOPMENT ########################
            # split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(train["pp_text"], train["target"])
            	
            
           	# create bag-of-words with weights using tfid vectoriser
           	# strip accents and remove stop words during vectorisation
            tf=TfidfVectorizer(strip_accents = 'ascii', stop_words='english')
            	
            
           	# transform and fit the training set with vectoriser
            X_train_tf = tf.fit_transform(X_train)
           	# transform the test set with vectoriser
            X_test_tf = tf.transform(X_test)
            	
            
            # create logistic regression model
            logreg = LogisticRegression(verbose=1, random_state=0, penalty='l2', solver='newton-cg')
           	# train model on  vectorised training data
            model = logreg.fit(X_train_tf, y_train)
           	# evaluate model performance on the test set
            pred = model.predict(X_test_tf)
            metrics.f1_score(y_test, pred, average='weighted')
            
            ################# EVALUATION ########################
            # importing the libraries
            
        	# converting the vectoriser and model into a pipeline
        	# this is necessary as LIME takes a model pipeline as an input
            c = make_pipeline(tf, model)
        	
        
        	# saving a list of strings version of the X_test object
            ls_X_test= list(X_test)
        	
        
        	# saving the class names in a dictionary to increase interpretability
            class_names = {0: 'non-disaster', 1:'disaster'}
        
            # create the LIME explainer
        	# add the class names for interpretability
            LIME_explainer = LimeTextExplainer(class_names=class_names)
        	
        
        	# choose a random single prediction
            idx = 15
        	# explain the chosen prediction 
        	# use the probability results of the logistic regression
        	# can also add num_features parameter to reduce the number of features explained
            LIME_exp = LIME_explainer.explain_instance(ls_X_test[idx], c.predict_proba)
        	# print results
            st.write('Document id: %d' % idx)
            st.write('Tweet: ', ls_X_test[idx])
            st.write('Probability disaster =', c.predict_proba([ls_X_test[idx]]).round(3)[0,1])
            st.write('True class: %s' % class_names.get(list(y_test)[idx]))
        
            # print class names to show what classes the viz refers to
            st.write("1 = disaster class, 0 = non-disaster class")
        	# show the explainability results with highlighted text
            html = LIME_exp.as_html()
            components.html(html, height=800)


