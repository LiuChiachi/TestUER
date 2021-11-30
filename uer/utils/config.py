import json
import sys
from argparse import Namespace


def load_hyperparam(default_args):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(default_args.config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)
    #print("config", config_args_dict)

    if "deepspeed" in default_args.__dict__:
        with open(default_args.deepspeed_config, mode="r", encoding="utf-8") as f:
            default_args.deepspeed_config_param = json.load(f)

    default_args_dict = vars(default_args)

    command_line_args_dict = {k: default_args_dict[k] for k in [
        a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)
    ]}
    #print("yesssssss1111", default_args_dict['emb_size'])
    default_args_dict.update(config_args_dict)
    #print("yesssssss22222", default_args_dict['emb_size'])
    default_args_dict.update(command_line_args_dict)
    #print("yesssssss3333", default_args_dict['emb_size'])
    args = Namespace(**default_args_dict)
    #print("yesssssss4444", args.emb_size)
    return args
