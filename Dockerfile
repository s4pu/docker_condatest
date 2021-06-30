FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Make sure the environment is activated:
#RUN echo "Make sure flask is installed:"
#RUN python -c "import flask"

COPY setup.py .
COPY run.py entrypoint.sh ./
COPY pyproject.toml .
COPY pycloud2room/ pycloud2room/
RUN python setup.py install
ENTRYPOINT ["./entrypoint.sh"]
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
#RUN python run.py
