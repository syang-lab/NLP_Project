import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import json

json_filename = '../data/raw_EDT/Trading_benchmark/evaluate_news.json'
with open(json_filename) as data_file:
    test_data = json.loads(data_file.read())

