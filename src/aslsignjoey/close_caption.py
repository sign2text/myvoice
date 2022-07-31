#%%
import cv2
#%%
class CloseCaption(object):
    FONT = cv2.FONT_HERSHEY_COMPLEX
    FONT_SCALE = 2.5
    # CV2 uses BGR Color
    FONT_COLOR = (0, 38, 178)
    THICKNESS = 3
    LINE_TYPE = cv2.LINE_AA
    TEXT_POSITION = (10, 220)
    MAX_WORDS = 10


    def __init__(self, frames:list, text:str, startat:float = 0.2,**kwargs):
        self.frames = frames
        self.text = text
        self.words = text.split() if text else []
        self.text_start_frame = int(len(frames) * startat) if startat < 1 and text else 0
        self.word_rate_frames = (len(frames) - self.text_start_frame) // (len(self.words) + 1) if text else 0
        self.FONT_COLOR = kwargs.get("font_color",(0,38,178))


    
    def get_frame(self):
        caption = ""
        for frame_index, frame in enumerate(self.frames):
            if self.text:
                num_words = (frame_index - self.text_start_frame) // self.word_rate_frames if self.word_rate_frames > 0 else len(self.words)
                if num_words > 0:
                    caption = " ".join(self.words[0:num_words][self.MAX_WORDS * -1 :])
                cv2.putText(frame, caption, self.TEXT_POSITION , self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.THICKNESS, self.LINE_TYPE)    
            yield frame

#%%
class CloseCaptionFile(CloseCaption):
    def __init__(self, file_name:str, text, 
             startat=0.2, 
             text_pos=(0.10,0.80), out_frame_size=None, fps=24, **kwargs):
        MIN_POS = 0.80
        cap = cv2.VideoCapture(file_name)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.TEXT_POSITION = (int(width * text_pos[0]), int(height * text_pos[1]))
        self.FPS = cap.get(cv2.CAP_PROP_FPS) if not fps else fps

        if out_frame_size is None:
            self.OUT_FRAME_SIZE = (int(width), int(height))
        else:
            self.OUT_FRAME_SIZE = out_frame_size
            self.FONT_SCALE = (width * height)/(out_frame_size[0]*out_frame_size[1])

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frames.append(frame)
        cap.release()
        super().__init__(frames, text, startat, **kwargs) 

    def write_video(self, file_name:str,codec:str = 'mp4v'):
        #Define the codec to use for writing video
        # List of fourcc codecs are listed at https://softron.zendesk.com/hc/en-us/articles/207695697-List-of-FourCC-codes-for-video-codecs#:~:text=List%20of%20FourCC%20codes%20for%20video%20codecs%20,8-bit%20Indexed%20Color%20%2068%20more%20rows%20
        fourcc = cv2.VideoWriter_fourcc(*codec)
        print(f" FPS: {self.FPS}\n OUT_FRAME_SIZE: {self.OUT_FRAME_SIZE}")
        mp4_out = cv2.VideoWriter(file_name,fourcc, self.FPS, self.OUT_FRAME_SIZE)
        for frame in self.get_frame():
            mp4_out.write(frame)
        mp4_out.release()
        

#%%

def test_classes():
    # test_list = [cv2.imread(f"../../data/raw/inp/images{i:04d}.png") for i in range(1,29)]
    # cc = CloseCaption(test_list,"Hello, how are you?")
    ccFile = CloseCaptionFile("../../data/raw/test/g1HvmBOR7Y4_3-3-rgb_front.mp4", "so i am going to do this")
    for i,f in enumerate(ccFile.get_frame()):
        cv2.imwrite(f"../../data/raw/out/images{i:04d}.png",f)
    ccFile.write_video("../../data/raw/out/g1HvmBOR7Y4_3-3-cc.mp4")
# %%
