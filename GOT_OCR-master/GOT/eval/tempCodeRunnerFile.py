
from transformers import TextStreamer
from GOT.model.plug.transforms import train_transform, test_transform
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html