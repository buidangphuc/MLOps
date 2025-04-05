FROM apache/airflow:2.10.5

ENV AIRFLOW_HOME=/opt/airflow

WORKDIR $AIRFLOW_HOME

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
         python3-pip \
         default-jdk \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && java -version

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin/:${PATH}"

COPY airflow.requirements.txt .

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r airflow.requirements.txt

COPY scripts scripts
RUN chmod +x scripts

USER airflow
