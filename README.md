# MLFacecut
a python script to prepare face images for machine learning

# Usage
Either load from another python script and use the run(...) method or call the script directly like 

`python prepare_images.py -i example1_small.jpg -f cnn -x "waifu2x-command-line-executeble" -w desiredXsize -h desiredYsize -o out.jpg`

# About the script
it uses opencv-python, face-recognition, so you might need to pip install those before usage.
Also face-recognition is using dlib which is GPU accelerated so install CUDA befor installing face-recognition.


The example face is from https://thispersondoesnotexist.com/
