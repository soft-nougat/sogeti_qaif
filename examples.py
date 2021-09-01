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
                titanic = sns.load_dataset('examples_data/titanic')
                
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
                
                data = pd.read_csv("examples_data/titanic.csv")
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
                      WEFE
                      """, 
                      alignment = "left")
    with col3:
        help.sub_text("""<b>Example dataset</b>""", 
                      alignment = "left")
        help.sub_text("""
                      <a href = https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>
                      TBD</a>""", 
                      alignment = "left")
    with col4:
        help.sub_text("""<b>Similar packages</b>""", 
                      alignment = "left")
        help.sub_text("""
                      TBD""", 
                      alignment = "left")
        
    st.markdown("""---""")
    
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
        here</a>. The demo is below. 
        """
        
        help.sub_text(exp_text)
        
        nbdt = open("demo_video/demo_nbdt.mp4", "rb")
        st.video(nbdt)
    
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
        
        button_xai_me = st.button('Run Example')
        
        if button_xai_me:
            with st.echo():
                
                import json
                import math
                from typing import Dict, Optional
                
                import matplotlib.pyplot as plt
                import numpy as np
                import shap
                
                from skimage.segmentation import slic
                from matplotlib.colors import LinearSegmentedColormap
                from keras.engine.training import Model
                from classification_models import Classifiers
                
                from coco_loader import CocoLoader
                
                IMG_COUNT = 100
                IMG_HEIGHT = 224
                IMG_WIDTH = 224
                IMG_CHANNELS = 3
                IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
                
                ResNet34, preprocess_input = Classifiers.get('resnet34')
                model = ResNet34(IMG_SHAPE, weights='imagenet')
                
                loader = CocoLoader()
                data = loader.load_sample(IMG_COUNT)
                data = preprocess_input(data)
                
                def get_class_names() -> Dict[int, str]:
                    url = "imagenet_class_index.json"
                    with open(url) as f:
                        return {int(k): v[1] for k, v in json.load(f).items()}
        
                def mask_image(
                    zs: np.array, 
                    segmentation: np.array, 
                    image: np.array, 
                    background: Optional[int] = None
                ) -> np.array:
                    if background is None:
                        background = image.mean((0,1))
                    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
                    for i in range(zs.shape[0]):
                        out[i,:,:,:] = image
                        for j in range(zs.shape[1]):
                            if zs[i,j] == 0:
                                out[i][segmentation == j,:] = background
                    return out
                
                def fill_segmentation(values: np.array, segmentation_lut: np.array) -> np.array:
                    out = np.zeros(segmentation_lut.shape)
                    for i in range(values.shape[0]):
                        out[segmentation_lut == i] = values[i]
                    return out
        
                def get_colormap() -> LinearSegmentedColormap:
                    blue = (22/255, 134/255, 229/255)
                    blue_shades = [(*blue, alpha) for alpha in np.linspace(1, 0, 100)]
                    red = (254/255 ,0/255, 86/255)
                    red_shades = [(*red, alpha) for alpha in np.linspace(0, 1, 100)]
                    return LinearSegmentedColormap.from_list("shap", blue_shades + red_shades)
                
                def plot_shap_top_explanations(
                    model: Model, 
                    image: np.array, 
                    class_names_mapping: Dict[int, str],
                    top_preds_count: int = 3,
                    fig_title: Optional[str] = None,
                    fig_name: Optional[str] = None
                ) -> None:
                    """
                    A method that provides explanations for N top classes.
                    :param model: Keras based Image Classification model
                    :param image: Single image in the form of numpy array. Shape: [224, 224, 3]
                    :param class_names_mapping: Dictionary that provides mapping between class inedex and name
                    :param top_preds_count: Number of top predictions that we want to explain
                    :param fig_title: Figure title
                    :param fig_name: Output figure path
                    :return:
                    """
                    
                    image_columns = 3
                    image_rows = math.ceil(top_preds_count / image_columns)
                    
                    segments_slic = slic(image, n_segments=100, compactness=30, sigma=3)
                    
                    def _h(z):
                        return model.predict(preprocess_input(mask_image(z, segments_slic, image, 255)))
                    
                    explainer = shap.KernelExplainer(_h, np.zeros((1,100)))
                    shap_values = explainer.shap_values(np.ones((1,100)), nsamples=1000)
                    
                    preds = model.predict(np.expand_dims(image, axis=0))
                    top_preds_indexes = np.flip(np.argsort(preds))[0,:top_preds_count]
                    top_preds_values = preds.take(top_preds_indexes)
                    top_preds_names = np.vectorize(lambda x: class_names[x])(top_preds_indexes)
                    
                    plt.style.use('dark_background')
                    fig, axes = plt.subplots(image_rows, image_columns, figsize=(image_columns * 5, image_rows * 5))
                    [ax.set_axis_off() for ax in axes.flat]
                    
                    max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
                    color_map = get_colormap()
                    
                    for i, (index, value, name, ax) in \
                        enumerate(zip(top_preds_indexes, top_preds_values, top_preds_names, axes.flat)):
                    
                        m = fill_segmentation(shap_values[index][0], segments_slic)
                        subplot_title = "{}. class: {} pred: {:.3f}".format(i + 1, name, value)
                        ax.imshow(image / 255)
                        ax.imshow(m, cmap=color_map, vmin=-max_val, vmax=max_val)
                        ax.set_title(subplot_title, pad=20)
                       
                    if fig_title:
                        fig.suptitle(fig_title, fontsize=30)
                    if fig_name:
                        plt.savefig(fig_name)
                    plt.show()
                
                image_example = data[2]
                class_names = get_class_names()
                plot_shap_top_explanations(model, image_example, class_names, top_preds_count=6, fig_title="SHAP", fig_name="viz/coco_resnet34_shap.png")
                
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


