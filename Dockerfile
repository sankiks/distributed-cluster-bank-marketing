FROM spark:4.2.0-python3

USER 0

RUN mkdir -p /opt/spark/code /opt/spark/data \
    && chown -R spark:spark /opt/spark/code /opt/spark/data



#path of the working directory


USER spark

WORKDIR /opt/spark/code