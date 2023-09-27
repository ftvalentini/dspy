from dsp.modules.hf_client import ChatModuleClient
from .signatures import *

from .retrieve import *
from .predict import *
from .primitives import *
# from .evaluation import *


# FIXME:


import dsp

settings = dsp.settings

Cohere = dsp.Cohere
OpenAI = dsp.GPT3
ColBERTv2 = dsp.ColBERTv2
Pyserini = dsp.PyseriniRetriever
ElasticSearch = dsp.ElasticSearch
HFClientTGI = dsp.HFClientTGI
ChatModuleClient = ChatModuleClient
