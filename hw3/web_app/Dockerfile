FROM python:3.11

RUN echo 'Launched build!'
WORKDIR /usr/src/app

RUN echo 'Copying project files'
ADD web_app /usr/src/app/web_app
COPY models.py /usr/src/app/web_app
COPY README.md /usr/src/app/
RUN chmod -R 777 /usr/src/app/

RUN echo 'Copying poetry files'
COPY poetry.lock pyproject.toml /usr/src/app/
RUN pip3 install poetry

RUN echo 'Installing dependencies'
RUN poetry install

RUN echo 'Exposing ports'
EXPOSE 5000

RUN echo 'Adding run command'
ENV RUNTIME_DOCKER Yes
CMD poetry run python web_app/app.py
