# Changes made to run model on Jetson  

1.  signjoey/model.py
Changed `tf.config.set_visible_devices([], "GPU")` to
```
if tf.__version__ >= '2.0':
    tf.config.set_visible_devices([], "GPU")
else:
    tf.compat.v1.config.experimental.set_visible_devices([],"GPU")
```  
Change was based on  https://github.com/tensorflow/tensorflow/issues/45954 

2. pip3 install --upgrade pip 
This was required because pip didn't find torchtext versions after 0.6.0

Below did not work
2.  python3 -m pip install torchtext=0.11.0
As per https://pypi.org/project/torchtext/#:~:text=Version%20Compatibility torchtext version for pytorch 1.9.0 is 0.10.0 However, install using `pip3 install torchtext=0.10.0` failed with `Could not find a version that satisfies the requirement torchtext==0.10.0 (from versions: 0.1.1, 0.2.0, 0.2.1, 0.2.3, 0.3.1, 0.4.0, 0.5.0, 0.6.0, 0.11.0, 0.11.1, 0.11.2)` 

Hence used 0.11.0 - and pray that it works

Google Collabe used pytorch 1.4.0 hence torchtext was 0.5.0. l4t-ml comes with torch 1.9.0

Running `pip3 install torchtext==0.11.0` gave `WARNING: pip is being invoked by an old script wrapper. This will fail in a future version of pip.`

changed torchtext.data to torchtext.legacy.data in following files 
if torchtext installed version is >= 0.9.0 
/w210/slt/signjoey/data.py
/w210/slt/signjoey/dataset.py
/w210/slt/signjoey/helpers.py 
/w210/slt/signjoey/prediction.py
/w210/slt/signjoey/training.py
/w210/slt/signjoey/vocabulary.py


