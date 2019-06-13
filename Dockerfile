# Use Ubuntu for our image
FROM ubuntu

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade

# Adding wget and bzip2
RUN apt-get install -y wget bzip2

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

# Configuring access to Jupyter
#RUN mkdir /opt/notebooks
WORKDIR /notebook
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

# Install flask and other libs
RUN pip install --upgrade pip
RUN pip install flask flask_cors jsonify numpy pandas
RUN pip install certifi chardet Click idna itsdangerous Jinja2 joblib MarkupSafe python-dateutil
RUN pip install pytz requests scikit-learn scipy six urllib3 Werkzeug 
#RUN pip install sklearn surprise scikit-surprise
RUN conda install -c conda-forge scikit-surprise

# Listens port
EXPOSE 5000

# Copying neccessary files into working folder
COPY ./p2-code /notebook

CMD ["python", "runserver.py"]
