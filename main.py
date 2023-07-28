import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# app = FaceAnalysis(providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')
# faces = app.get(img)
# rimg = app.draw_on(img, faces)
# print('writing')
# cv2.imwrite("./.output/t1_output.jpg", rimg)

from gfpgan.utils import GFPGANer
from pathlib import Path
model_path = str(Path("./models/GFPGANv1.4.pth"))
# todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
GFPGANer(
    model_path=model_path,
    arch="clean",
    channel_multiplier=2,
    upscale=1,
    device="cpu",
)