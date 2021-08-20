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
import streamlit.components.v1 as components
import pandas as pd
import base64
from PIL import Image
import helper as help
        
# app setup
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
                                 'Model Validation',
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
             
        elif step == 'Data Understanding':
            
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
             
        elif step == 'Data Preparation':
            
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
        
        elif step == 'Model Development':
            
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
             
        elif step == 'Model Validation':
            
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
        
        elif step == 'Model Deployment':
            
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
    
    elif section == 'Technical examples':
        
        # Side panel setup
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Sogeti-logo-2018.svg/1280px-Sogeti-logo-2018.svg.png", 
                         use_column_width=True)
        
        help.header("Technical Examples",
                    is_sidebar = False)
        
        step = st.sidebar.radio("Select Problem statement",
                                ('Dataset bias',
                                 'Model interpretability', 
                                 'Model adequacy',
                                 'AI Model version control',
                                 'Data version control'))
        
        if step == 'Dataset bias':
            
            help.set_bg_hack('gate2_bg.png')
            
            bias_g1 = """
             Understanding bias in the dataset and the model is the first step in creating a fair ML model. 
             After bias is detected, bias correction methods should be chosen accordingly. 
             Once the bias is mitigated, metrics can be calculated to test if the dataset is fair.
             """
              
            help.expander('Understanding the problem',
                          bias_g1)
            
            bias_g2 = """
             Use stastical methods (EDA) to detect: Sampling bias, group attribution bias, 
             omitted variable bias. 
             Use correction methods like: Sampling and re-weighting. 
             To correct bias in the model use: adversarial debiasing, reject-object classification.
             """
            
            help.expander('Data Understanding',
                          bias_g2)
            
            expander = st.beta_expander('Data Preparation', expanded=False)
    
            with expander:
                
                exp_text = """
                The fairlearn.metrics module provides the means to assess fairness-related 
                metrics for models. This applies for any kind of model that users may already 
                use, but also for models created with mitigation techniques from the 
                Mitigation section. The Fairlearn dashboard provides a visual way to 
                compare metrics between models as well as compare metrics for 
                different groups on a single model.
                <p> Recall score for ungrouped metrics </p>
                A measure of whether the model finds all the positive cases in the 
                input data. The scikit-learn package implements this in 
                sklearn.metrics.recall_score().
                """
                
                help.sub_text(exp_text)
                
                with st.echo():
                    
                    # Load all necessary packages
                    import sys
                    sys.path.insert(1, "../")  
                    
                    import numpy as np
                    np.random.seed(0)
                    
                    from aif360.datasets import GermanDataset
                    from aif360.metrics import BinaryLabelDatasetMetric
                    from aif360.algorithms.preprocessing import Reweighing
                    
                    from IPython.display import Markdown, display
                    
                    dataset_orig = GermanDataset(
                    protected_attribute_names=['age'],           # this dataset also contains protected
                                                                 # attribute for "sex" which we do not
                                                                 # consider in this evaluation
                    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
                    features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
                    )
                    
                    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
                    
                    privileged_groups = [{'age': 1}]
                    unprivileged_groups = [{'age': 0}]
                    
                    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
                    display(Markdown("#### Original training dataset"))
                    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
                    
                    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
                    dataset_transf_train = RW.fit_transform(dataset_orig_train)
                    
                    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
                    display(Markdown("#### Transformed training dataset"))
                    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
            
            
    elif section == 'Blogs':
        
        help.set_bg_hack('gate2_bg.png')
        
        link = '[Programming Fairness into your ML model by Almira Pillay](https://medium.com/sogetiblogsnl/programming-fairness-into-your-machine-learning-model-a3a5479bfe41)'
        st.markdown(link, unsafe_allow_html=True)
        
        
except TypeError:
     st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
