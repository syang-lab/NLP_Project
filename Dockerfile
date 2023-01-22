FROM python:3.8.15

############# Install Python packages ############
COPY ./requirements.txt /opt/ml/model/requirements.txt

RUN pip install --upgrade pip==22.3.1
RUN pip install --requirement /opt/ml/model/requirements.txt

############# Copying Sources ##############
COPY FTE_NLP/data/raw_EDT/Event_detection /opt/ml/model/FTE_NLP/data/raw_EDT/Event_detection
COPY FTE_NLP/model /opt/ml/model/FTE_NLP/model
COPY FTE_NLP/train_eval_pred /opt/ml/model/FTE_NLP/train_eval_pred
COPY FTE_NLP/utils /opt/ml/model/FTE_NLP/utils
COPY FTE_NLP/configs /opt/ml/model/FTE_NLP/configs
COPY FTE_NLP/experiments/test /opt/ml/model/FTE_NLP/experiments/test

WORKDIR /opt/ml/model/


# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# the Python script that should be invoked and used as the entry point for training.
ENTRYPOINT  ["python", "FTE_NLP/train_eval_pred/domain_adaption_main.py", \
        "--train_eval_config_dir_filename", "FTE_NLP/configs/domain_adaption_train_eval_config.yaml",\
        "--train_eval_data_directory", "FTE_NLP/data/raw_EDT/Event_detection/dev_test.json" ,\
        "--cloud_directory", "project-nlp-375001-aiplatform"]
