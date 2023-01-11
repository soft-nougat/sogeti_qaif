# Sogeti's Quality AI Framework

This is the web app that will be used to present Sogeti's golden standard in delivering quality AI solutions. The app has been developed in Streamlit, the API docs of which can be found [HERE](https://docs.streamlit.io/en/stable/api.html).

## Getting started

Build a docker image using:

```docker build . -t qaif```

Then, to run a container using this image:

```docker run --name qaif -p 8501:8501 qaif```

For development purposes, you might want to build a dev image containing dependencies and mount the sourcecode to the container, such that the docker image has access to recent changes:

```docker build . -f Dockerfile.dev  -t qaif-dev```
```docker run --name qaif-dev -v `pwd`:/qaif -p 8501:8501 qaif-dev```




Alternatively, if you want to run the app without using Docker:

To install the required packages:

```pip install -r requirements.txt```

Then, to get the app up and running locally:

```streamlit run qaif.py```

The output should be something like:

```
You can now view your Streamlit app in your browser.

Network URL: http://172.69.255.255:8501
External URL: http://83.85.122.99:8501
```
 
 Enter the network URL from **your** terminal window in your browser address bar to view the app.



