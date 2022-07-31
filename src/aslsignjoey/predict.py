"""Inference module for translating videos

Translate a video using the best model obtained during training using slt/signjoey. 

Example Usage
-------------
For running inference, pass the config used for training and the file to translate
```
# To get translated text
python3 predict.py translate configs/wlasl_gls.yaml data/infer_in/61916.mp4

# To write a caption file out, to get output in different folder
python3 predict.py caption configs/wlasl_gls.yaml data/infer_in/61916.mp4 --output data/infer_out

# To write a caption file out, to get output in same folder as 61916_caption.mp4
python3 predict.py caption configs/wlasl_gls.yaml data/infer_in/61916.mp4 
```

If inference is run on a different device than training, the following files are needed:  
ckpt file -  the checkpoint from the step producing the best result
gls.vocab and txt.vocab


"""
#%%
import aslutils  
ROOT = aslutils.set_path()
#%%
import argparse
import yaml
from pathlib import Path
from pprint import pprint, pformat

import torch

from packaging import version 
from torchtext import __version__ as torchtext_ver
if version.parse(torchtext_ver) >= version.parse("0.9.0"):
    from torchtext.legacy import data
else:
    from torchtext import data

from aslsignjoey.data_format import format_video
from aslsignjoey.data_features import get_tensor
from aslsignjoey.datasets import ASLDataset
from aslsignjoey.close_caption import CloseCaptionFile

from signjoey.batch import Batch
from signjoey.data import make_data_iter
from signjoey.model import build_model, SignModel
from signjoey.vocabulary import GlossVocabulary, TextVocabulary, Vocabulary
from signjoey.prediction import validate_on_data
from signjoey.vocabulary import ( 
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)

#%%
logger, model, config, use_cude, device = None, None, None, None, None
#%%
def translate(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input file {args.input} not found") 
    setup(args)

    fields = get_fields()
    data =  get_filedataset(fields, args.input)

    valid_iter = make_data_iter(
        dataset=data,
        batch_size=32,
        batch_type=config['batch_type'],
        shuffle=False,
        train=False,
    )
    logger.debug(f"Length iter {len(valid_iter)} Iterator {str(valid_iter)}")

    i = 0
    model.eval()
    with torch.no_grad():
        for valid_batch in iter(valid_iter):
            i += 1
            logger.debug(f"Processing iteration {i}")
            sgn, sgn_lengths = valid_batch.sgn
            logger.debug(f"Sign lengths:{sgn_lengths} and Sign:{sgn.shape}")
            batch = Batch(
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=config['txt_pad_index'],
                sgn_dim=config['feature_size'],
                use_cuda=use_cuda,
                frame_subsampling_ratio=None,
            )
            (
                batch_gls_predictions,
                batch_txt_predictions,
                batch_attention_scores,
            ) = model.run_batch(
                batch=batch,
            )    
    txt_trans_arr = model.txt_vocab.arrays_to_sentences(arrays=batch_txt_predictions)
    txt_trans = [" ".join(t) for t in txt_trans_arr]
    logger.info(f"Translation:{txt_trans[0]}")

    return txt_trans[0]

#%%
def caption(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input file {args.input} not found") 
    # First get the translated text
    logger.debug(f"Processing captions for {args.input}")
    text = translate(args)
    cap_cls = CloseCaptionFile(args.input, text)
    if args.output:
        if os.path.isdir(args.output):
            out_file = os.path.join(args.output, os.path.basename(args.input))
        else:
            out_file = args.output 
    else:
        out_file = aslutils.get_outfilename(args.input,suffix="_caption")
    cap_cls.write_video(out_file)
    logger.info(f"Caption file written to {out_file} text:[{text}].")

#%%
def setup(args):
    global logger, model, config 
    if logger is None: 
        logger = aslutils.get_logger(verbose=args.verbose)
    logger.debug(args)
    if model is None:
        config, model = get_model(args)

#%%

def get_model(args):
    global use_cuda, device
    assert os.path.isfile(args.config), f"Configuration file {args.config} not found"


    cfg = aslutils.load_config(args.config)
    infer = {**cfg["data"], **cfg["testing"], **cfg["training"]}
    logger.debug(pformat(infer))
    do_recognition = infer.get("recognition_loss_weight",1.0) > 0.0
    do_translation = infer.get("translation_loss_weight",1.0) > 0.0
    feature_size = infer.get("feature_size")
    batch_type = infer.get("batch_type", "sentence")

    ckpt_file = get_ckpt_file(args.ckpt,infer.get("model_dir",None))


    #Let the config overwrite CUDA use if one exists. 
    # However if CUDA is not available, don't use value from config
    device, use_cuda = aslutils.get_device(infer.get("use_cuda",True)) 

    # Vocabulary would be saved as a part of training. Download the file and specify location
    gls_vocab, txt_vocab = load_vocab(infer)
    txt_pad_index = txt_vocab.stoi[PAD_TOKEN]

    model = build_model(
        cfg=cfg["model"],
        gls_vocab = gls_vocab,
        txt_vocab = txt_vocab,
        sgn_dim = feature_size,
        do_recognition = do_recognition,
        do_translation = do_translation)
    model_checkpoint = torch.load(ckpt_file, map_location = device)
    model.load_state_dict(model_checkpoint["model_state"])
    model.to(device)

    config = {'do_recognition' : do_recognition,
              'do_translation' : do_translation,
              'feature_size'   : feature_size,
              'batch_type'     : batch_type,
              'txt_pad_index'  : txt_pad_index
             }

    logger.info(model)
    return (config, model)

#%%
def get_fields(pad_feature_size:int = 1024,txt_lowercase: bool = True):
    sequence_field = data.RawField()
    signer_field = data.RawField()
    # Don't know why the fields need to be tagged this way
    # For now, just copy these from slt
    sgn_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        preprocessing=tokenize_features,
        tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((pad_feature_size,)),
    )

    return (sequence_field, signer_field, sgn_field)

def tokenize_features(features):
    ft_list = torch.split(features, 1, dim=0)
    return [ft.squeeze() for ft in ft_list]

# NOTE (Cihan): The something was necessary to match the function signature.
def stack_features(features, something):
    return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

#%%
def get_filedataset(fields, input_file):
    out_file = aslutils.get_outfilename(input_file,suffix="_fmt")
    use_file = out_file
    logger.debug(f"Refomatting {input_file} into {use_file} ")

    result = format_video(input_file, out_file)
    if result:
        logger.warn(f"Formatting failed for {input_file} with {result[1]}:{result[2]}")
        use_file = input_file

    f, sign = get_tensor(use_file)

    logger.debug(f"Received tensor {sign.shape} for {use_file}")
    if use_file != input_file:
        os.remove(use_file)
    ds = ASLDataset(sign, fields, logger=logger)
    return ds
#%%
def get_ckpt_file(ckpt:str, alt_dir:str) -> str:
    
    #First check if the best.ckpt exists in the model dir
    #From training, best.ckpt is a symbolic link to the best checkpoint
    # So it needs to be resolved to the actual file name to use

    # Note about file size calculation below
    # Find file size in MB. 
    # Each right shift divides by 2, hence 20 right shifts = 2^20 resulting in MB 
    ckpt_file = None
    if ckpt and os.path.isfile(ckpt):
        ckpt_file = ckpt
    if not ckpt_file:
        if not os.path.isdir(alt_dir):
            alt_dir = str(ROOT / alt_dir)
        best_file = os.path.join(alt_dir,'best.ckpt')
        best_file_size = 0
        if os.path.isfile(best_file):
            ckpt_file = Path(best_file).resolve()
            best_file_size = os.path.getsize(best_file) >> 20
            logger.debug(f"Best file {best_file} resolved to {ckpt_file} "
                         f"Filesize: {os.path.getsize(ckpt_file) >>20}MB")
        if best_file_size < 1:
            #Find the latest modified file among all the checkpoint files
            ckpt_file = max(aslutils.get_files(alt_dir,"[0-9]*.ckpt"), key=os.path.getctime)
    logger.debug(f"Using checkpoint file {ckpt_file}")
    if ckpt and ckpt != ckpt_file:
        logger.warn(f"Checkpoint file specified {ckpt} not found. Using {ckpt_file} instead")
    if ckpt_file:
        fsize = os.path.getsize(ckpt_file) >> 20
        if fsize < 1: 
            logger.warn(f"Checkpoint file {ckpt_file} is smaller than 1MB")
    else:
        raise FileNotFoundError(f"Could not find checkpoint file in {alt_dir} or {ckpt}")
    return ckpt_file


#%%
def load_vocab(cfg: dict) -> (Vocabulary,Vocabulary):

    model_dir = cfg.get("model_dir",None)
    if model_dir and not os.path.isdir(model_dir) :
        model_dir = str(ROOT / model_dir)

    # Find if vocabulary files are specially configured.
    # if not specifically configured, look for the vocabs 
    # in the location where the training would have stored them.
    gls_vocab_file = cfg.get("gls_vocab", None)
    logger.debug(f"gls_vocab_file from config {gls_vocab_file}. Model dir {model_dir}")
    if gls_vocab_file and not os.path.isfile(gls_vocab_file):
        gls_vocab_file = str(ROOT / gls_vocab_file)
    elif model_dir:
        gls_vocab_file = os.path.join(model_dir, "gls.vocab")

    logger.debug(f"Final gls_vocab_file {gls_vocab_file}")

    txt_vocab_file = cfg.get("txt_vocab", None)
    if txt_vocab_file and not os.path.isfile(txt_vocab_file):
        txt_vocab_file = str(ROOT / txt_vocab_file)
    elif model_dir:
        txt_vocab_file = os.path.join(model_dir, "txt.vocab")

    logger.debug(f"Final txt_vocab_file {txt_vocab_file}")

    gls_vocab = GlossVocabulary(file=gls_vocab_file)
    txt_vocab = TextVocabulary(file=txt_vocab_file)

    return gls_vocab, txt_vocab


#%%
def parse_args(inline_options = None, known=False):
    """
        inline_options = ['caption',
                          'configs/wlasl_gls.yaml',
                          'data/infer_in/61916.mp4',
                          '--output','data/infer_out',
                          '--verbose']
    """
    parser = argparse.ArgumentParser('ASL Inference')
    parser.add_argument('mode',type=str, choices=["translate","caption"], 
                 help='Translate or close caption the file')
    parser.add_argument('config',type=str, help='Config file to use')
    parser.add_argument('input',type=str, help='File to process')
    parser.add_argument('--ckpt',type=str, help='Checkpoint file to use')
    parser.add_argument('--output',type=str, default=None, 
                 help="File to write output to. If empty and caption is chosen,"
                      " caption file written as _caption")
    parser.add_argument('-v','--verbose', action='store_true')

    if known is None:
        args = parser.parse_known_args(inline_options)[0] if inline_options else parser.parse_known_args()[0]
    else:
        args = parser.parse_args(inline_options) if inline_options else parser.parse_args()
    return args

#%%
if __name__ == "__main__":
    #Required when running in interactive session. 
    # Should be changed to False before running in batch scripts, otherwise parameters specified with spelling errors may just be ignored
    args = parse_args() 
    logger = get_logger(args.verbose)
    if args.mode == "translate":
        translate(args)
    elif args.mode == "caption":
        caption(args)