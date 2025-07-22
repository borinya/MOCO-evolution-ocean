# model_factory.py
from .transformer_models import OceanTransformer, Informer, Autoformer

def get_model(args):
    model_map = {
        'Transformer': OceanTransformer,
        'Informer': Informer,
        'Autoformer': Autoformer
    }
    return model_map[args.model_type](args)