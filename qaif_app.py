# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 08:58:11 2021

@author: TNIKOLIC

QAIF app for dissemination of knowledge on best practices in developing AI models 
and ethical requirements

"""

# newest release has the improves session state
#  st.session_state.counter = 0

import streamlit as st
from PIL import Image
import helper as help
import principles 
        
# app setup
st.set_option('deprecation.showPyplotGlobalUse', False)
try:
    
    # Main panel setup
    help.header('Sogeti Quality AI Framework',
                is_sidebar=False)
    
    help.sub_text('QAIF app for dissemination of knowledge on best practices in developing AI models and ethical considerations')
    
    section = st.selectbox("Choose topic", 
                            ("Theoretical basis", 
                             "Technical examples",
                             "Blogs"))
    
    if section == "Theoretical basis":
    
        # Side panel setup
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Sogeti-logo-2018.svg/1280px-Sogeti-logo-2018.svg.png", 
                         use_column_width=True)
        
        help.header("QAIF Gates",
                    is_sidebar=True)
        
        step = st.sidebar.radio("Select Gate to read more information",
                                ('High Level information',
                                 'Business Understanding', 
                                 'Data Understanding',
                                 'Data Preparation',
                                 'Model Development',
                                 'Model Evaluation',
                                 'Model Deployment'))
        
        if step == 'High Level information':
            
            help.header("Theoretical basis",
                        is_sidebar = False)
             
            help.set_bg_hack('main_bg.png')
            
            main_text = """
            Inspired by the CRISP-DM Framework, <b>QAIF</b> is a cohesive, generic framework that 
            can be tailored to a specific AI solution in a given context. 
            The framework is comprised of six gates that follow the process flow of the 
            AI project development cycle (CRISP-DM). The gates can be broken down into project phase, 
            processes, outcomes, governance and people. In each gate, there are specific tasks that need 
            to be completed for the gate to be passed through in order to enter the next gate.
            This ensures that each phase of the AI development cycle is validated thoroughly.
            <br> Please select the block/gate in the left sidebar to read more specific info. </br>
            <br> </br>
            """
            
            help.sub_text(main_text)
           
            image = Image.open('qaif_main.png')
            st.image(image, caption='The blocks and corresponding gates of the QAIF')
            
            image_1 = Image.open('qaif_secondary.png')
            st.image(image_1)
        
        elif step == 'Business Understanding':
            
            help.header("Business Understanding",
                        is_sidebar = False)
            
            help.set_bg_hack('gate1_bg.png')
            
            gate1_text = """
            In order to pass Gate 1, the <b>Business Understanding</b> phase needs to be completed by 
            determining the approach in which the business problem will be resolved. 
            In this phase, the tasks of identifying stakeholders, product requirement specifications, 
            technical design specifications, performance metrics and ethical/legal compliance will 
            be completed for the process of AI model development to run smoothly and efficiently. 
            <br> The outcomes of this gate are data specifications that will be used in Gate 2. </br> 
            <br> </br>
            """
            
            help.sub_text(gate1_text)
            
            image = Image.open('gate1_main.png')
            st.image(image, caption='The info on the first gate')
            
            principles.bu_principles()
             
        elif step == 'Data Understanding':
            
             help.header("Data Understanding",
                        is_sidebar = False)
            
             help.set_bg_hack('gate2_bg.png')
             
             gate2_text = """
                <b>Data Understanding</b> phase defines Gate 2, bringing specifiations from the first gate 
                and domain knowledge and experience together in order to  understand inherent biases 
                or assumptions of the data this solution will be dealing with. In this phase, 
                exploratory data analysis (EDA) is performed through methods like statistical parity 
                and many more, results of which will be presented in form of a data description report. 
                The data description report will aid the model development team in determining the 
                initial model and the metrics it will be evaluated on.  
                <br> The outcomes of this gate are data specifications that will be used in Gate 3. </br> 
                <br> </br>
                """
              
             help.sub_text(gate2_text)
                 
             image = Image.open('gate2_main.png')
             st.image(image, caption='The info on the second gate')
             
             principles.du_principles()
             
        elif step == 'Data Preparation':
            
             help.header("Data Preparation",
                        is_sidebar = False)
            
             help.set_bg_hack('gate3_bg.png')
             
             gate3_text = """
             Bringing the insights from the previous gate, the <b>Data Preparation</b> phase can begin, 
             where the data engineering team, domain experts and model developers play crucial roles. 
             Tasks like data mining, data quality assessment and training data construction define 
             this phase’s process. The use of synthetic data is advocated when there is senstive 
             information or the dataset needs to be boosted. 
             <br>The outcome of this gate is high 
             quality training data for the model, which enables the next phase of model development to begin.
             </br>
             <br> </br>
             """
              
             help.sub_text(gate3_text)
             
             image = Image.open('gate3_main.png')
             st.image(image, caption='The info on the third gate')
             
             principles.dp_principles()
             
        
        elif step == 'Model Development':
            
             help.header("Model Development",
                        is_sidebar = False)
            
             help.set_bg_hack('gate4_bg.png')
             
             gate4_text = """
             The <b>AI Model Development</b> phase starts when Gate 3 is opened, 
             with high quality training data. 
             The main role in this phase is played by the model developers, who ensure that the AI model 
             they are developing is precise and works with the data prepared in phases 2 and 3. 
             To be sure of this, performance metrics are drawn from the model and presented to 
             the stakeholders. Furthermore, one of the tasks is testing of model performance and 
             functionality on the most granular level. 
             <br>The outcome of this gate is a trained model, which can be validated in next gate.
             </br>
             <br> </br>
             """
              
             help.sub_text(gate4_text)
             
             image = Image.open('gate4_main.png')
             st.image(image, caption='The info on the fourth gate')
             
             principles.md_principles()
             
        elif step == 'Model Evaluation':
            
             help.header("Model Evaluation",
                        is_sidebar = False)
            
             help.set_bg_hack('gate5_bg.png')
             
             gate5_text = """
             In Gate 5, we enter the <b>Model Evaluation</b> phase. 
             As the model’s already been validated on the most granular level in the previous phase, 
             this gate’s tasks focus on ensuring that the model is transparent and works according 
             to the business ethical considerations set in the Business Understanding phase. 
             Being the most important phase in the QAIF, it ensures that the AI model is 
             fair and understandable. The people included in this process are testers, 
             developers and the legal team. The outcome is the quality review and explainable AI 
             result of the model. 
             <br>The outcome of this gate is a validated model, which can be deployed in next gate.
             </br>
             <br> </br>
             """
              
             help.sub_text(gate5_text)
             
             image = Image.open('gate5_main.png')
             st.image(image, caption='The info on the fifth gate')
             
             principles.me_principles()
        
        elif step == 'Model Deployment':
            
             help.header("Model Deployment",
                         is_sidebar = False)
            
             help.set_bg_hack('gate6_bg.png')
             
             gate6_text = """
             Once we have a transparent and understandable model, we can enter the 6th and final phase 
             – <b>Deployment</b>. This phase focuses on further testing of the AI model on a higher level,
             ensuring it works in accordance to the process and architecture. 
             Furthermore, this phase’s tasks include collecting final documentation of the model for 
             auditing and maintenance. With the final phase coming to an end, we have checked all 
             the boxes and passed through all the gates. 
             <br>If it turns out this model needs to be 
             retrained or adjusted, we can always turn back and revisit any of the phases, as the 
             AI project life cycle is an iterative process.</br>
             <br> </br>
             """
              
             help.sub_text(gate6_text)
             
             image = Image.open('gate6_main.png')
             st.image(image, caption='The info on the sixth gate')
             
             principles.d_principles()
    
    elif section == 'Technical examples':
        
        # Side panel setup
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Sogeti-logo-2018.svg/1280px-Sogeti-logo-2018.svg.png", 
                         use_column_width=True)
        
        help.header("Technical Examples",
                    is_sidebar = True)
        
        example = st.sidebar.radio("Select Problem statement",
                                    ('Dataset bias',
                                     'Model interpretability', 
                                     'Model adequacy',
                                     'AI Model version control',
                                     'Data version control'))
        
        if example == 'Dataset bias':
            
            help.set_bg_hack('gate2_bg.png')
            
            help.header("Dataset Bias",
                        is_sidebar = False)
        
            data = st.sidebar.radio("Select Data Type",
                                    ('Tabular',
                                     'Text', 
                                     'Images'))
            
            bias_g1 = """
             <p><b>Introduction</b></p>
             Imbalanced datasets pose a challenge for training predictive models as most of 
             the algorithms to train these models are not designed to handle class imbalance. 
             This leads to poor predictive results, as the rules learned from the data by 
             the model are not representative for the real world or the broader picture. 
             Most AI algorithms learn a mapping between the input-data and the prediction. 
             When the classes presented to the model are imbalanced, it can not accurately 
             learn an accurate mapping, especially towards minority classes. 
             Essentially, you could say that a bias inthe data generally results in a 
             biased model. This bias is a threat to external validity – it limits 
             the generalizability of your findings to a broader group of people, 
             which in many cases is not justundesirable for the accuracy of your model 
             but also creates unwanted breaches of ethicalstandards.
             In practice, not detecting and dealing with class imbalance could 
             for example result in modelsthat are racist, sexist or discriminate 
             based on other sensitive properties such as age, religionand gender 
             among other factors. For instance, think of a model that because of 
             skewed dataunfairly predicts people with certain skin colours to have 
             a higher rate of recidivism. Or take 
             <a href="https://algorithmwatch.org/en/google-vision-racism/">this example</a>
             where Google's computer vision AI produced labels starkly different depending
             on skintone on given images.
             On this page we first explain potential causes of class imbalance, followed by 
             a brief exampleand methods to deal with this problem adequately.
             <p><b>Causes of Class Imbalance</b></p>
             Class imbalance can be caused by multiple factors.
             One of them simplymeasurement errors which cause a deviation between the
             A sample is a subset of individuals from a larger population. This allows 
             you to learn thecharacteristics of a population if the sample actually 
             reflects the population.
             biased sampling, measurement errors
             people with specific characteristics are more likely toagree to take part in the study.
             Sampling because large set
             Sampling because subset of data fortraining
             """
              
            help.expander('Understanding the problem',
                          bias_g1)
            
            if data == 'Tabular':
                
                tabular_intro = """
                <p>Keywords: Class imbalance, Normalization, Sampling, Stratisfied, Bias, Tabular data</p>
                <p>Packages used: Pandas, SKLearn</p>
                <p>Example dataset: <a href = https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>Titanic seaborn</a></p>
                <p>Similar packages to prevent class imbalance: Fairlearn</p>
                """
                
                help.sub_text(tabular_intro)
                
                expander = st.beta_expander('Data Understading', 
                                            expanded=False)
        
                with expander:
                    
                    exp_text = """
                    At this stage it is important to get insight in the data you are going to use. 
                    Detect potential vulnerable variables.
                    Note: it could be that certain values of variables in your data have strong 
                    predictive power, however that this is unwanted or unethical for the task at 
                    hand as well. E.g. an automated model
                    for insurance approval could implicitily learn that someone with an age over 30 
                    has a higher
                    chance of being accepted than someone below the age of 30 with an exact 
                    similar situation.
                    Measuring disparity in predictions is further handled and explained in: 
                    "Model adequacy".
                    In the upcoming example we load the commonly-known titanic dataset. 
                    This data consists of
                    information of all passengers that embarked the titanic. Information such as: 
                    name, ticket, boat,age, sex, room and ticket-type are present. 
                    A full overview of the variables are shown below.
                    The goal in this example is to predict the survival status of individual 
                    passengers on the Titanic  after training on the (sampled) dataset and 
                    see what the effect of the sampling has on the model accuracy.
                    """
                    
                    help.sub_text(exp_text)
                    
                    button_du = st.button('Run Data Understanding Example')
                    
                    if button_du:
                        # only execute this code when expanded + clicked
                        with st.echo():
                            import seaborn as sns
                            #import matplotlib.pyplot as plt
                            titanic = sns.load_dataset('titanic')
                            
                            x = titanic['sex'].value_counts()
                                
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
                            #sns.catplot(x ="sex", hue ="survived", kind ="count", data = titanic)
                            #sns.factorplot("class", data=titanic, hue="sex")
                
                
                expander = st.beta_expander('Data Preparation', 
                                            expanded=False)
        
                with expander:
                    
                    exp_dp_text = """
                    Let's say we would like to have a sample from this dataset for training, 
                    then we could randomly sample 
                    Class normalization
                    Sampling: stratisfied - preserve original population
                    Adjusting weights
                    Estimate missing data of classes
                    Define a target population and a sampling frame 
                    (the list of individuals that the sample will be
                    drawn from). Match the sampling frame to the target population 
                    as much as possible to reduce the risk of sampling bias.
                    Oversampling can be used to avoid sampling bias in situations where members of defined
                    groups are underrepresented (undercoverage). This is a method of selecting respondents from
                    some groups so that they make up a larger share of a sample than they actually do the
                    population.
                    After all data is collected, responses from oversampled groups are weighted to their
                    actual share of the population to remove any sampling bias.
                    Stratified random sampling is one common method that is used by researchers because it
                    enables them to obtain a sample population that best represents the entire population being
                    studied, making sure that each subgroup of interest is represented.
                    """
                    
                    help.sub_text(exp_dp_text)
                    
                    help.sub_text("""
                                  First we have to do some standard preprocessing of the data so we can work with it and it is
                                  ready to be inserted into a model. This entails filling in the missing values, removing noninformative
                                  variables and basic one-hot-encoding for categorical variables. Details of the
                                  processing are left out of this notebook for now but similar steps were taken in the following
                                  URL. For now we just import the preprocessed dataset with similar sex-ratio's as previously
                                  explored.
                                  """)
                    
                    button_dp = st.button('Run Data Preparation Example')
                    
                    if button_dp:
                        # only execute this code when expanded + clicked
                        with st.echo():
                            
                            import pandas as pd
                            from sklearn.model_selection import train_test_split
                            data = pd.read_csv("titanic.csv")
                            #del data['Unnamed: 0']
                            X, y = data.iloc[:, 1:], data.iloc[:, 0]
                            # Random sampling - split the dataset in train and test - force low amount of women
                            X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd = train_test_split(X, y, test_size=0.33, random_state=42)
                            st.write(str(X_train_rnd['sex'].value_counts()['female']))
                            
            if data == 'Text':
                    
                text_intro = """
                <p>Keywords: Class imbalance, Normalization, Sampling, Stratisfied, Bias, Text data</p>
                <p>Packages used: WEFE</p>
                <p>Example dataset: <a href = https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>TBD</a></p>
                """
            
                help.sub_text(text_intro)
                
                expander = st.beta_expander('Data Understading', 
                                            expanded=False)
        
                with expander:
                    
                    exp_text = """
                    Word embeddings are dense vector representations of words trained from document corpora. 
                    They have become a core component of natural language processing (NLP) downstream systems 
                    because of their ability to efficiently capture semantic and syntactic relationships
                    between words. A widely reported shortcoming of word embeddings is that they are prone 
                    to inherit stereotypical social biases exhibited in the corpora on which they are trained.
                    """
                    
                    help.sub_text(exp_text)
                        
                    
                    
            
    elif section == 'Blogs':
        
        help.set_bg_hack('gate2_bg.png')
        
        link = '[Programming Fairness into your ML model by Almira Pillay](https://medium.com/sogetiblogsnl/programming-fairness-into-your-machine-learning-model-a3a5479bfe41)'
        st.markdown(link, unsafe_allow_html=True)
        
        
except TypeError:
     st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
