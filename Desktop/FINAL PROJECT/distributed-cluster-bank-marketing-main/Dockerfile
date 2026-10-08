FROM spark:4.2.0-python3

# Switch to root
USER 0

# Create directories and set permissions
RUN mkdir -p /opt/spark/code /opt/spark/data \
    && chown -R spark:spark /opt/spark/code /opt/spark/data

# Install numpy
RUN pip install numpy

# Copy dataset files
COPY bank_master.csv bank_worker1.csv bank_worker2.csv /opt/spark/data/
RUN chown -R spark:spark /opt/spark/data/*

# Switch to spark user
USER spark

# Set working directory
WORKDIR /opt/spark/code