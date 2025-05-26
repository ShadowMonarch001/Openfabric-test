FROM openfabric/tee-python-cpu:dev

RUN pip install --upgrade openfabric_pysdk

# Copy only necessary files for Poetry installation
RUN apt-get update && apt-get install -y build-essential cmake

COPY pyproject.toml ./

# Install dependencies using Poetry
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade poetry && \
    python3 -m poetry install --only main && \
    rm -rf ~/.cache/pypoetry/{cache,artifacts}

RUN mkdir -p /models

RUN pip install --upgrade openfabric_pysdk

RUN mkdir -p /openfabric/app/result

RUN pip install sentence-transformers faiss-cpu



# Copy the rest of the source code into the container
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 8888

# Start the Flask app using the start.sh script
CMD ["sh","start.sh"]