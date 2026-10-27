import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [BASE_DIR, os.path.join(BASE_DIR, "generative-discriminative-lab")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image, ImageDraw
from backend.preprocessing import preprocess_image
from backend.inference import load_or_train_classifier, classify_digit

cnn, acc = load_or_train_classifier()
test_cases = {
    '1': [(100, 30), (100, 170)],
    '7': [(50, 40), (150, 40), (80, 170)],
    '0': [(100, 40), (50, 100), (100, 160), (150, 100), (100, 40)],
    '8': [(100, 40), (130, 70), (70, 130), (100, 160), (130, 130), (70, 70), (100, 40)],
    '4': lambda d: (d.line([(130, 30), (130, 170)], fill=255, width=18), d.line([(130, 130), (50, 130), (100, 40)], fill=255, width=18))
}

for name, draw_fn in test_cases.items():
    img = Image.new('L', (200, 200), color=0)
    draw = ImageDraw.Draw(img)
    if callable(draw_fn):
        draw_fn(draw)
    else:
        draw.line(draw_fn, fill=255, width=18)
    cnn_t, _, disp = preprocess_image(img)
    res = classify_digit(cnn, cnn_t)
    print(f"True: {name} -> Predicted: {res['predicted_digit']} ({res['confidence']*100:.1f}%)")
