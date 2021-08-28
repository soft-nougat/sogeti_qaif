# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 08:37:51 2021

@author: TNIKOLIC
"""
import helper as help

def bu_principles():
    
    traceability = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we define how inputs will be traced along the development cycle.
    It is important to understand this in order to have a clear overview of where the
    data comes from and how it ends up.</br>
    <b>Practical guide</b>
    <br>Implement an MLOps solution to monitor and trace inputs and outputs like 
    <dfn title="MLflow is an open source platform to manage the ML lifecycle. 
    It includes a tracking component."> MlFlow.</dfn></br>
    <b>Packages</b>
    <br><a href = 'https://mlflow.org/'>MLFlow</a></br>
    '''
    
    help.expander('Traceability',
                   traceability)
    
    auditability = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we define who will be able to access the inputs and source code.
    Especially in the case of <dfn title= "The term “PII,” as defined in OMB Memorandum M-07-1616 
    refers to information that can be used to distinguish or trace an individual's identity, 
    either alone or when combined with other personal or identifying information that is linked or 
    linkable to a specific individual.">sensitive (PII)</dfn> applications.</br>
    <b>Practical guide</b>
    <br>Enable security authentication or encryption frameworks.</br>
    <b>Packages</b>
    <br>Enable On-premise or cloud authentication service like VPNs or Active directory 
    (Azure cloud) in development VM. Furthermore, package 
    <a href = 'https://docs.python.org/3/library/hashlib.html'>Hashlib</a> 
    can be used for secure encryption.</br> 
    '''
    
    help.expander('Auditability',
                   auditability)
    
    fairness = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we define fairness metrics and detection methods if 
    UC applies to sensitive groups.</br>
    <b>Practical guide</b>
    <br>Use methods like Stastical Parity difference; Equal Opportunity Difference; Average Odd difference ; Disparate Impact. </br>
    <b>Packages</b>
    <br><a href = ' https://github.com/fairlearn/fairlearn/tree/main/fairlearn/metrics'>Fairlearn</a>
    is a Python package that empowers developers of artificial intelligence (AI) systems to assess 
    their system's fairness and mitigate any observed unfairness issues. Fairlearn contains mitigation
    algorithms as well as metrics for model assessment.</br>
    '''
    
    help.expander('Fairness',
                   fairness)
    
    transparency = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we define transparency methods throughout the lifecycle based on 
    end-user requirements </br>
    <b>Practical guide</b>
    <br>Implement explainable AI methods to understand the outcome of the model.</br>
    <b>Packages</b>
    <br><a href = 'https://github.com/marcotcr/lime'>Lime</a> &
    <a href = 'https://github.com/slundberg/shap'>Shap</a></br>
    '''
    
    help.expander('Transparency',
                   transparency)
    
    robustness = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we create a definition of ready and define performance metrics to 
    ensure the model is robust.</br>
    <b>Practical guide</b>
    <br>Implement performance tests (data input size, API calls etc.); data drift testing methods; unit testing.</br>
    <b>Packages</b>
    <br>See other phases for packages and methods.</br>
    '''
    
    help.expander('Robustness',
                   robustness)
    
    privacy  = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we assess the GDPR impact the project will have.</br>
    <b>Practical guide</b>
    <br>Develop mitigation strategy for data pre-processing and post-processing phase.</br>
    <b>Packages</b>
    <br>See data preparation phase.</br>
    '''
    
    help.expander('Privacy',
                   privacy)
    
def du_principles():
    
    traceability = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we review data collection, inputs and state 
    of data.</br>
    <b>Practical guide</b>
    <br> Implementing a <dfn title= "Data Version Control is a new type of 
    data versioning, workflow, and experiment management software, that builds 
    upon Git (although it can work stand-alone). DVC reduces the gap between 
    established engineering tool sets and data science needs, allowing users 
    to take advantage of new features while reusing existing skills and intuition.">
    data version control system (DVC).</dfn>
    Set a documented specification with info on where the data is located, 
    what kind of source it comes from, responsible people, any privacy 
    or quality concerns.</br>
    <b>Packages</b>
    <br><a href = 'https://dvc.org/doc/install'>DVC</a></br>
    '''
    
    help.expander('Traceability',
                   traceability)
    
    auditability = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we review DVC outputs, pinpoint data handling steps and validate them.</br>
    <b>Practical guide</b>
    <br>Audit checks (peer review). <dfn title = "A technical review on the data sources and data 
    handling brings the quality of said sources to a higher level."> Techincal audit checks </dfn>, the auditer needs to understand the code.</br>
    <b>Packages</b>
    <br>Provide EDA of initial data and manipulated data - tf data validation (see below) 
    in a dashboard or a document.</br> 
    '''
    
    help.expander('Auditability',
                   auditability)
    
    fairness = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we review data collection methods; check data adequacy 
    (e.g. missing data for certain groups) and bias.</br>
    <b>Practical guide</b>
    <br>Bias detection using EDA methods and 
    <a href = "https://share.streamlit.io/soft-nougat/dqw-ivves/app.py">DQW</a>.</br>
    <b>Packages</b>
    <br><a href = 'https://www.tensorflow.org/tfx/data_validation/get_started'>TF Data Validation</a>;
    <a href = 'https://github.com/pandas-profiling/pandas-profiling'>Pandas Profiling</a>;
    <a href = 'https://pypi.org/project/sweetviz/ '>SweetViz</a>;
    <a href = 'https://share.streamlit.io/soft-nougat/dqw-ivves/app.py'>DQW</a>
    </br>
    '''
    
    help.expander('Fairness',
                   fairness)
    
    transparency = '''
    <b>Theoretical guide</b>
    <br>The first 2 steps provide checkpoints for transparency - making sure the data sources 
    are traceable and of high quality.</br>
    <b>Practical guide</b>
    <br>This step should include another auditer, from the business side. Not technical. 
    Should recieve a dashboard or extensive documentation from step 1 & 2, review and approve/decline.</br>
    '''
    
    help.expander('Transparency',
                   transparency)
    
    robustness =  '''
    <b>Theoretical guide</b>
    <br>
    <li>1 - Making sure the data source contains data suitable for model training and testing. </li>
    <li>2 - Making sure production data is never used directly in development. </li>
    <b>Practical guide</b>
    <br>
    <li>1 - Separating training and testing data into different tables and environments to prevent data leaks.</li>
    <li>2 - Separating prod, dev, acc data</li>
    <b>Packages</b>
    <br>
    <li>1 - Preparing separate data tables.</li>
    <li>2 - ETL strategies in each environment.</li>
    '''
    
    help.expander('Robustness',
                   robustness)
    
    privacy  = '''
    <b>Theoretical guide</b>
    <br> Understand senstive data inputs for GDPR purposes. </br>
    <b>Practical guide</b>
    <br> Sogeti's PII identifier & ADA </br>
    <b>Packages</b>
    <br> <a href = "https://pypi.org/project/piidetect/"> PII Detect </a> </br>
    '''
    
    help.expander('Privacy',
                   privacy)
    
    
def dp_principles():
    
    traceability = '''
    <b>Theoretical guide</b>
    <br>Under this principle, we prepare data pipelines in model development.</br>
    <b>Practical guide</b>
    <br> Implementing a <dfn title= "The data pipeline would ensure traceability 
    because it includes explicit steps that read data from source and transformed it.">
    data pipeline.</dfn> </br>
    <b>Packages</b>
    <br><a href = 'https://www.tensorflow.org/guide/data'>TF Data pipeline</a></br>
    '''
    
    help.expander('Traceability',
                   traceability)
    
    auditability = '''
    <b>Theoretical guide</b>
    <br>DVC solution for versioning and monitoring.</br>
    <b>Practical guide</b>
    <br><dfn = "For auditors, the main driver of using data analytics is to improve audit quality. 
    It allows auditors to more effectively audit the large amounts of data held and processed in 
    IT systems in larger clients."> Audit checks </dfn> after the data pipeline step can include
    an DVC versioning output with an EDA of data used for the training of the model.</br>
    '''
    
    help.expander('Auditability',
                   auditability)
    
    fairness = '''
    <b>Theoretical guide</b>
    <br>Review bias correction methods .</br>
    <b>Practical guide</b>
    <br>Sampling, Reweighting , Feature engineering, Synthetic data.</br>
    <b>Packages</b>
    <br><a href = 'https://github.com/Trusted-AI/AIF360'>Trusted AI</a>
    </br>
    '''
    
    help.expander('Fairness',
                   fairness)
    
    transparency = '''
    <b>Theoretical guide</b>
    <br>Transparent data processing pipeline.</br>
    <b>Practical guide</b>
    <br>Display results of processing methods (dashboard).</br>
    <b>Packages</b>
    <br><a href = 'https://pypi.org/project/sweetviz/'>SweetViz</a>
    </br>
    '''
    
    help.expander('Transparency',
                   transparency)
    
    robustness =  '''
    <b>Theoretical guide</b>
    <br>
    <li>Mutation testing for model stability and robustness </li>
    <li>Data pre-processing pipeline checks</li>
    <b>Practical guide</b>
    <br>Create Data mutations.</br>
    <b>Packages</b>
    <br><a href = "https://pypi.org/project/MutPy/"> MutPy </a></br>
    '''
    
    help.expander('Robustness',
                   robustness)
    
    privacy  = '''
    <b>Theoretical guide</b>
    <br> Determine how to handle senstive inputs; Differential privacy. </br>
    <b>Practical guide</b>
    <br> Generate synthetic/mask data; Including smart noise to data. </br>
    <b>Packages</b>
    <br> <a href = "https://github.com/joke2k/faker"> Faker </a> 
    <a href = "https://github.com/opendp/smartnoise-sdk"> Smartnoise </a> </br>
    '''
    
    help.expander('Privacy',
                   privacy)
    
def md_principles():
    
    traceability = '''
    <b>Theoretical guide</b>
    <br>Under this principle, it is important to ensure model development is traceable.
    To accomodate this an MLOPs solution for versioning and monitoring is used. </br>
    <b>Practical guide</b>
    <br> Use <dfn title= "MLflow is an open source platform to manage the ML lifecycle, 
    including experimentation, reproducibility, deployment, and a central model registry.">
    MlFlow </dfn>for model versioning and log output.</br>
    <b>Packages</b>
    <br><a href = 'https://mlflow.org/'>MlFlow</a></br>
    '''
    
    help.expander('Traceability',
                   traceability)
    
    auditability = '''
    <b>Theoretical guide</b>
    <br>MLOPs solution for versioning and monitoring, technical peer review on the model 
    version used prior to pushing to development.</br>
    <b>Practical guide</b>
    <li> 1. Use ML Flow for model versioning and log output to collect information. </li>
    <li> 2. Create repos, pipelines and split into dev(test), acceptance, production. 
    Add required revewers on each step making sure only the administrators can change reviewers, 
    not developers or business users. </li>
    <li> 3. Use a git version control system to push a new version of the code to different environments.</li>
    </br>
    '''
    
    help.expander('Auditability',
                   auditability)
    
    fairness = '''
    <b>Theoretical guide</b>
    <br>Assess if the model is biased during development.</br>
    <b>Practical guide</b>
    Use bias mitigation methods like:
    <li> Adversarial debiasing </li>
    <li> Reject-object classification </li>
    <b>Packages</b>
    <br><a href = 'https://github.com/Trusted-AI/AIF360'>Trusted AI</a>
    </br>
    '''
    
    help.expander('Fairness',
                   fairness)
    
    transparency = '''
    <b>Theoretical guide</b>
    <br>Is the output of the model easily understandable to the developer?</br>
    <b>Practical guide</b>
    <br> Use Explainable AI techniques. </br>
    <b>Packages</b>
    <br><a href = 'https://github.com/marcotcr/lime'>Lime</a> &
    <a href = 'https://github.com/slundberg/shap'>Shap</a></br>
    '''
    
    help.expander('Transparency',
                   transparency)
    
    robustness =  '''
    <b>Theoretical guide</b>
    <br>Assess if model decisions are influenced by irrelevant factors.</br>
    <b>Practical guide</b>
    <li>Deploy model <dfn title="The degradation of model performance due to changes in 
    data and relationships between input and output variables.">drift</dfn> 
    detection methods.</li>
    <li> Unit tests. </li>
    <b>Packages</b>
    <br><a href = "https://github.com/SeldonIO/alibi-detect">Alibi-Detect</a></br>
    '''
    
    help.expander('Robustness',
                   robustness)
    
    privacy  = '''
    <b>Theoretical guide</b>
    <br> Monitoring access to the development environment and
    hosting environments for GDPR compliancy . </br>
    <b>Practical guide</b>
    <br> Use a VPN tunnel and authentication. </br>
    '''
    
    help.expander('Privacy',
                   privacy)
    
def me_principles():
    
    traceability = '''
    <b>Theoretical guide</b>
    <br>Under this principle, it is important to assess if the previous phases'
    traceability principles were followed. </br>
    <b>Practical guide</b>
    <br> User assesses reports/dashboards/logs and accepts/declines changes.</br>
    '''
    
    help.expander('Traceability/Auditability',
                   traceability)
    
    
    fairness = '''
    <b>Theoretical guide</b>
    <br>User should assess fairness reports from gates 3 and 4.</br>
    '''
    
    help.expander('Fairness',
                   fairness)
    
    transparency = '''
    <b>Theoretical guide</b>
    <br>Evaluation against acceptance criteria set in business understanding.</br>
    <b>Practical guide</b>
    <br>Execute acceptance tests (XAI output of model development in a report format). </br>
    '''
    
    help.expander('Transparency',
                   transparency)
    
    robustness =  '''
    <b>Theoretical guide</b>
    <br>Model KPI assessment - the KPIs are set in business understanding.</br>
    <b>Practical guide</b>
    <li>User acceptance tests (XAI output of model development in a report format)</li>
    <li>Adversarial attacks and metamorphic tests.</li>
    '''
    
    help.expander('Robustness',
                   robustness)
    
    privacy  = '''
    <b>Theoretical guide</b>
    <br> Evaluation against senstive data handling KPIs set in business understanding. </br>
    <b>Practical guide</b>
    <br>It is important to review all privacy-related data and model augmentation 
    that was done in phase 2-4. Provide a report for the stakeholders to review and 
    share with privacy officers.</br>
    '''
    
    help.expander('Privacy',
                   privacy)

def d_principles():
    
    traceability = '''
    <b>Theoretical guide</b>
    <br>Use an MLOPs solution for production monitoring and usage.
    Following previous gate principles makes sure the model can be 
    deployed and maintained in production.</br>
    '''
    
    help.expander('Traceability/Auditability',
                   traceability)
    
    
    transparency = '''
    <b>Theoretical guide</b>
    <br>Making sure the model is used what it was built for.</br>
    <b>Practical guide</b>
    <br>Sessions with users to transfer knowledge about the process and model.
    Diligent documentation.</br>
    '''
    
    help.expander('Fairness/Transparency',
                   transparency)
    
    robustness =  '''
    <b>Theoretical guide</b>
    <br> Track robustness KPIs set in gate 1. </br>
    <b>Practical guide</b>
    <br>Set KPI alerts on monitoring dashboards following model drift and other metrics.</br>
    '''
    
    help.expander('Robustness',
                   robustness)
    
    privacy  = '''
    <b>Theoretical guide</b>
    <br>Ensuring the right people have access to model and dashboards.</br>
    <b>Practical guide</b>
    <br>Authentication keys.</br>
    '''
    
    help.expander('Privacy',
                   privacy)