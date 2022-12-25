from transformers import AutoModelForMaskedLM


model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)