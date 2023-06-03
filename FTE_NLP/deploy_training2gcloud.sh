#!/usr/bin/env bash
# create bucket #
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export BUCKET_NAME=${PROJECT_ID}-aiplatform
echo $BUCKET_NAME

export REGION=us-west1
gsutil mb -l $REGION gs://$BUCKET_NAME


# build docker #
export IMAGE_REPO_NAME=nlp_image
export IMAGE_TAG=01.2023_V0
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
echo $IMAGE_URI
docker build -f Dockerfile -t $IMAGE_URI ./


# push docker #
gcloud auth configure-docker
docker push $IMAGE_URI


# submit job #
export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --model-dir=$BUCKET_NAME \

# monitor job #
gcloud ai-platform jobs describe $JOB_NAME
gcloud ai-platform jobs stream-logs $JOB_NAME
