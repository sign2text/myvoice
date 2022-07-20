#%%
import os

#%%
def get_outfile(orgfile:str, suffix:str) -> str:
    """Appends suffix to the filename, while preserving extension correctly"""
    return os.path.join(os.path.dirname(orgfile),
                        os.path.basename(orgfile).replace(".",suffix + "."))

