From python:3.10
RUN mkdir qaif

COPY . /qaif

WORKDIR /qaif

RUN pip3 install -r requirements.txt
RUN curl https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py > /usr/local/lib/python3.10/site-packages/google/protobuf/internal/builder.py
CMD ["streamlit", "run", "/qaif/qaif_app.py"]
