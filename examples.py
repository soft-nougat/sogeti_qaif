# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 10:15:12 2021

@author: TNIKOLIC
"""

import streamlit as st
import helper as help

def tabular_bias():
    
    help.header("Tabular Data",
                is_sidebar = False)
    
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
                      Pandas
                      <br>SKLearn""", 
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
                      Fairlearn""", 
                      alignment = "left")
        
    expander = st.beta_expander('Data Understading', 
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
        ready to be inserted into a model.For now we just import the preprocessed dataset with 
        similar sex-ratio's as previously explored.
        """
        
        help.sub_text(exp_dp_text)
        
        button_dp = st.button('Run Data Preparation Example')
        
        if button_dp:
            # only execute this code when expanded + clicked
            with st.echo():
                
                import pandas as pd
                from sklearn.model_selection import train_test_split
                
                data = pd.read_csv("titanic.csv")
                #del data['Unnamed: 0']
                X, y = data.iloc[:, 1:], data.iloc[:, 0]
                
                # Random sampling - split the dataset in train and test - force low amount 
                # of women in trainset
                X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd = train_test_split(X, 
                                                                                    y, 
                                                                                    test_size=0.33, 
                                                                                    random_state=42)
                
                st.write(X_train_rnd['Sex'].count())

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
            
            data = pd.read_csv("titanic.csv")
            #del data['Unnamed: 0']
            X, y = data.iloc[:, 1:], data.iloc[:, 0]
            
            # Random sampling - split the dataset in train and test - force low amount 
            # of women in trainset
            X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd = train_test_split(X, 
                                                                                y, 
                                                                                test_size=0.33, 
                                                                                random_state=42)
            # only execute this code when expanded + clicked
            with st.echo():
                from sklearn.tree import DecisionTreeClassifier

                def accuracy_score(y_pred, y_test):
                    a = list(y_pred)
                    b = list(y_test)
                    accuracy = len([a[i] for i in range(0, len(a)) if a[i] == b[i]]) / len(a)
                    print(accuracy)
                
                clf = DecisionTreeClassifier()
                clf = clf.fit(X_train_rnd,y_train_rnd)
                
                y_pred = clf.predict(X_test_rnd)
                st.write("Accuracy:", accuracy_score(y_test_rnd, y_pred))
                
