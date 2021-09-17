from PIL import Image
import helper as help
import principles 
import examples

def render_app(streamlit):
    r = Renderer(streamlit)

    r.render_app()

class Renderer:
    streamlit = False

    def __init__(self, streamlit):
        self.streamlit = streamlit

    def render_app(self):
        st = self.streamlit

        # Main panel setup
        help.header('Sogeti Quality AI Framework',
                is_sidebar=False)
    
        help.sub_text('QAIF app for dissemination of knowledge on best practices in developing AI models and ethical considerations')
    
        section = st.selectbox("Choose topic", 
                            ("Theoretical basis", 
                             "Technical examples",
                             "Blogs"))
    
        if section == "Theoretical basis":

            self.render_section_theoretical_basis()
        
        elif section == 'Technical examples':

            self.render_section_technical_examples()
                
        elif section == 'Blogs':

            self.render_section_blogs()


    def render_section_theoretical_basis(self):
        st = self.streamlit

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

            self.render_step_high_level_information()
        
        elif step == 'Business Understanding':
            
            self.render_step_business_understanding()
            
        elif step == 'Data Understanding':

            self.render_step_data_understanding()
            
        elif step == 'Data Preparation':

            self.render_step_data_preparation()
        
        elif step == 'Model Development':

            self.render_step_model_development()
            
        elif step == 'Model Evaluation':

            self.render_step_model_evaluation()
        
        elif step == 'Model Deployment':
            
            self.render_step_model_deployment()


    def render_step_high_level_information(self):
        st = self.streamlit

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
    
        image = Image.open('bg_other/qaif_main.png')
        st.image(image, caption='The blocks and corresponding gates of the QAIF')
        
        image_1 = Image.open('bg_other/qaif_secondary.png')
        st.image(image_1)


    def render_step_business_understanding(self):
        st = self.streamlit

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
        
        image = Image.open('bg_other/gate1_main.png')
        st.image(image, caption='The info on the first gate')
        
        principles.bu_principles()


    def render_step_data_understanding(self):
        st = self.streamlit

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
            
        image = Image.open('bg_other/gate2_main.png')
        st.image(image, caption='The info on the second gate')
        
        principles.du_principles()
    

    def render_step_data_preparation(self):
        st = self.streamlit

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
        
        image = Image.open('bg_other/gate3_main.png')
        st.image(image, caption='The info on the third gate')
        
        principles.dp_principles()


    def render_step_model_development(self):
        st = self.streamlit

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
        
        image = Image.open('bg_other/gate4_main.png')
        st.image(image, caption='The info on the fourth gate')
        
        principles.md_principles()

    def render_step_model_evaluation(self):
        st = self.streamlit

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
        
        image = Image.open('bg_other/gate5_main.png')
        st.image(image, caption='The info on the fifth gate')
        
        principles.me_principles()


    def render_step_model_deployment(self):
        st = self.streamlit

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
        
        image = Image.open('bg_other/gate6_main.png')
        st.image(image, caption='The info on the sixth gate')
        
        principles.d_principles()


    def render_section_technical_examples(self):
        st = self.streamlit

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

            self.render_example_dataset_bias()
                        
        elif example == 'Model interpretability':
            
            self.render_example_model_interpretability()

        elif example == 'Data version control':

            self.render_example_data_version_control()

    
    def render_example_dataset_bias(self):

        help.set_bg_hack('gate2_bg.png')
            
        help.header("Dataset Bias",
                    is_sidebar = False)
    
        data = self.render_dt_sidebar()
        
        bias_g1 = """
            <b>The problem</b>
            <br><span style = "color:#F26531">
            <dfn title = "A disproportionate weight in favor of or against an idea or thing, usually in a way that is closed-minded, prejudicial, or unfair.">
            Bias</dfn></span> is a threat to external validity of the model – 
            it limits the generalizability of your findings to a broader group of people, 
            but also creates unwanted breaches of ethical standards.
            <br><b>The practicality</b>
            <br>In practice, not detecting and dealing with class imbalance could 
            for example result in models that are racist, sexist or discriminate. 
            For instance, take <a href="https://algorithmwatch.org/en/google-vision-racism/">
            this example</a> where Google's computer vision AI produced labels starkly 
            different depending on skintone on given images.
            <br><b>Causes of Class Imbalance</b>
            <br>Class imbalance can be caused by multiple factors:</br>
            <li><span style = "color:#F26531">
            <dfn title = "The difference between the observed value of a variable and the true, but unobserved, value of that variable.">
            - Measurement errors</dfn></span></li>
            <li><span style = "color:#F26531">
            <dfn title = "A sampling method is called biased if it systematically favors some outcomes over others.">
            - Biased sampling</dfn></span></li>
            """
        
        help.expander('Understanding the problem',
                    bias_g1)
        
        if data == 'Tabular':
            
            examples.tabular_bias()
                        
        if data == 'Text':
                
            examples.text_bias()

    
    def render_example_model_interpretability(self):
        help.set_bg_hack('gate1_bg.png')
            
        help.header("Model interpretability",
                    is_sidebar = False)

        data = self.render_dt_sidebar()
        
        xai_g1 = """
            Model transparency is key imperative in a business context. 
            The users need to understand why a model reaches certain decisions without having 
            to look into the code.
            <span style = "color:#F26531">
            <dfn title =
            "Explainable AI aims to mimic model behaviour at a global (explanations of how 
            the model works from a general point of view) and/or local level (explanations 
            of the model for a sample) to help explain how the model came to its decision.">
            Explainable AI </dfn></span> is an important component in creating production ready machine 
            learning models, as it supports user requirements for interpretability and transparency.
            """
        
        help.expander('Understanding the problem',
                    xai_g1)
        
        if data == "Tabular":
        
            examples.tabular_xai()
            
        elif data == "Images":
            
            examples.image_xai()
            
        elif data == "Text":
            
            examples.text_xai()
    

    def render_example_model_adequacy(self):
        pass
    

    def render_example_ai_model_version_control(self):
        pass
    

    def render_example_data_version_control(self):
        help.set_bg_hack('gate1_bg.png')
            
        help.header("Data version control",
                    is_sidebar = False)

        use_case_summary = """
            Developing an accurate model requires a lot of tweaking of parameters, training, 
            checking your output, tweaking again, training again, checking again, etc. Until we find one or more 
            configurations that meet our requirements within an acceptable error margin. The challenge here is 
            to make our results reproducible, so we can run a specific version of our model with specific 
            parameters and be sure that our output is the same every time. 
            <br><br>
            This is where data version control comes in. Built upon the same principles as \"regular\" version 
            control (e.g. Git), it allows us to save snapshots of our configurations, data, and model in a repository, as well as 
            effortlessly switch between those versions.
            """

        help.expander('Understanding the problem', use_case_summary)

        examples.dvc()


    def render_dt_sidebar(self):
        return self.streamlit.sidebar.radio("Select Data Type",
                                ('Tabular',
                                 'Text', 
                                 'Images'))

    def render_section_blogs(self):
        st = self.streamlit

        help.set_bg_hack('gate2_bg.png')
    
        link = '[Programming Fairness into your ML model by Almira Pillay](https://medium.com/sogetiblogsnl/programming-fairness-into-your-machine-learning-model-a3a5479bfe41)'
        st.markdown(link, unsafe_allow_html=True)
