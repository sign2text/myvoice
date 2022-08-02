from aslutils import get_logger
import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, RawField
from typing import Tuple

# Ideally we would have extended from signjoey.SignTranslationDataset 
# in case signjoey required specific class
# However, signjoey.SignTranslationDataset uses init to load from file
# So start from scratch
class ASLDataset(data.Dataset):
    def __init__(
        self,
        source,
        fields: Tuple[RawField,RawField,Field,Field,Field],
        has_target: bool = False,
        logger = None,
        **kwargs
    ):
        """ Create a Dataset for Translation dataset 
            
            :param source: Can be either video embeddings
            :param fields: A tuple containing the fields that will be used for translation
            :param has_target: If True, translations must exist in the file. 
                Note: has_target cannot be true if source is not a pickle file
            :param **kwargs: Remaining keyword arguments: Passed to the constructor of
                    data.Dataset.
        """
        if not logger:
            self.logger = get_logger()
        else:
            self.logger = logger
        if not isinstance(fields[0], (tuple, list)):
            self.logger.debug(f"Instance of fields[0] is {type(fields[0])}")
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                # ("gls", fields[3]),
                # ("txt", fields[4]),
            ]
        

        examples = []
        if isinstance(source,torch.Tensor) or \
           (isinstance(source, list) and isinstance(source[0], torch.Tensor)):
            if has_target:
                self.logger.warn("Has_Target was {has_target} but tensor supplied. Parameter ignored") 
            examples = self._load_tensor(source, fields)
        
        super().__init__(examples, fields, **kwargs)
    
    def _load_tensor(self, source:torch.Tensor, fields):
        examples = []
        examples.append(
            data.Example.fromlist(
                [
                    [f"predict-key-{i}" for i in range(len(source))],
                    [f"predict-signer-{i}" for i in range(len(source))],
                    # This is for numerical stability
                    source + 1e-8,
                    # ".",
                    # ".",
                ],
                fields,
            )
        )
        return examples


