import datetime
from google.cloud import storage


def save_model(model_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    project_id = "project-nlp-375001"
    bucket = storage.Client(project_id).bucket(model_dir)
    blob = bucket.blob('{}/{}'.format(datetime.datetime.now().strftime('nlp_%Y%m%d_%H%M%S'), model_name))
    blob.upload_from_filename(model_name)