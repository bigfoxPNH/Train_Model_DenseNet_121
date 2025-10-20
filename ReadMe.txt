#install venv
python -m venv venv
#activate venv
.\venv\Scripts\activate
#install requirements.txt
pip install -r requirements.txt

Crop:
    Run:    python skull_crop.py -i preprocess -o cropped_out --hough
        or  python skull_crop.py -i preprocess -o cropped_out --hough --scale_axes 1.15 --pad_ratio 0.1 
    Suggest value:
        scale_axes: 1.25–1.40
        pad_ratio: 0.10–0.18

Quality:
    Run: python enhance_quality.py -i cropped_out -o enhanced_images