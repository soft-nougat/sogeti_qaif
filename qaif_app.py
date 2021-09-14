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
from render import render_app
        
# app setup
st.set_option('deprecation.showPyplotGlobalUse', False)

try:
    render_app(st)    
        
except TypeError:
     st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
