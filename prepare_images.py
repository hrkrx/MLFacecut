import argparse
import os
import cv2
import dlib
import subprocess
import face_recognition
show_result = False

def get_image(args):
    image = cv2.imread(args["image"])
    return image

def get_face(image, override = False):
    if dlib.DLIB_USE_CUDA is False or override == "hog":
        print ("using hog")
        face_locations = face_recognition.face_locations(image, model="hog")

    elif override == "csc-face":
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags = cv2.CASCADE_SCALE_IMAGE
        )
        face_locations = [(f[1], f[0]+f[2], f[1]+f[3], f[0]) for f in faces]

    elif override == "csc-face-alt":
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags = cv2.CASCADE_SCALE_IMAGE
        )
        face_locations = [(f[1], f[0]+f[2], f[1]+f[3], f[0]) for f in faces]

    elif override == "csc-eyes-glasses":
        faceCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags = cv2.CASCADE_SCALE_IMAGE
        )
        face_locations = [(f[1], f[0]+f[2], f[1]+f[3], f[0]) for f in faces]

    elif override == "csc-eyes":
        eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        eyes = eye_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags = cv2.CASCADE_SCALE_IMAGE
        )
        print(eyes)
        face_locations = [(f[1], f[0]+f[2], f[1]+f[3], f[0]) for f in eyes]
    elif override == "facenet":
        face_locations = []
    else:
        print ("using cnn")
        face_locations = face_recognition.face_locations(image, model="cnn") # for better face-recognition 
    
    return face_locations

def get_square(image, face):
    (H, W) = image.shape[:2] # get image size
    (top, right, bottom, left) = face # top == Y1, right == X2, bottom == Y2, left == X1
    mid_point = ((int((right - left) / 2 + left)), (int((bottom - top) / 2 + top)))
    distances = [mid_point[0], W - mid_point[0],  mid_point[1], H - mid_point[1]]
    min_distance = min(distances)
    x1 = mid_point[0] - min_distance
    x2 = mid_point[0] + min_distance
    y1 = mid_point[1] - min_distance
    y2 = mid_point[1] + min_distance
    print ((top, right, bottom, left))
    if show_result is True:
        radius = 20
        color = (255, 0, 0)
        color2 = (0, 255, 0)
        thickness = 2
        image_copy = image.copy()
        image_copy = cv2.circle(image_copy, mid_point, radius, color, thickness)
        image_copy = cv2.rectangle(image_copy, (left, top), (right, bottom), color, thickness)
        image_copy = cv2.rectangle(image_copy, (x1, y1), (x2, y2), color2, thickness)
        cv2.imshow("Center", image_copy)
        cv2.waitKey(0)

    return (x1, y1, x2, y2)

def denoise_and_scale(image, width, height, waifu_executable):
    image_copy = image.copy()
    (H, W) = image_copy.shape[:2]

    if waifu_executable != None and H < height and W < width:
        if os.path.isdir("temp") == False:
            os.mkdir("temp")
        cv2.imwrite("temp/waifu.jpg", image_copy)
        file_path = os.path.abspath("temp/waifu.jpg")
        arguments = "-i \"{}\" -m noise_scale --scale_ratio 2 --noise_level 1".format(file_path)
        command = "{} {}".format(waifu_executable, arguments)
        print(command)
        result = subprocess.call(command, shell=True)
        if result != 0:
            print("Error in waifu2x call:", result)
       
        image_copy = cv2.imread("temp/waifu(CUnet)(noise_scale)(Level1)(x2.000000).png")
        os.unlink("temp/waifu(CUnet)(noise_scale)(Level1)(x2.000000)waifu.png")

    return cv2.resize(image_copy, (width, height))

def main(args):
    image = get_image(args)
    face_locations = get_face(image, args["force_model"])

    if show_result is True:
        print(face_locations)

    if len(face_locations) < 1 and args["force_model"] != "cnn":
        print("No face was detected")

        #TODO: Add rotation testing
        
        print("using cnn, this might take a while")
        face_locations = get_face(image, "cnn")
        
    
    if len(face_locations) > 0:
        if len(face_locations) > 1:
            print("Multiple faces found:", len(face_locations), ". using only the first")
        face = face_locations[0]
        print("Face found at:", face)
        img_rect = get_square(image, face)
        print("Cropping:", img_rect)
        image_cropped = image.copy()[img_rect[1]:img_rect[3], img_rect[0]:img_rect[2]]
        if show_result:
            cv2.imshow("cropped", image_cropped)
            cv2.waitKey(0)
        
        final_image = denoise_and_scale(image_cropped, args["width"], args["height"], args["waifu_executable"])
        cv2.imwrite(args["out_file"], final_image)

    else:
        print("No face found")

def run(input_image, width=512, height=512, out_file="out.jpg", force_model="hog", waifu_executable=None):
    args = [input_image, width, height, out_file, force_model, waifu_executable]
    print(args)
    main(args)
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
        help="path to input image")
    ap.add_argument("-y", "--width", type=int, default=512,
        help="resized image width (should be multiple of 32)")
    ap.add_argument("-x", "--height", type=int, default=512,
        help="resized image height (should be multiple of 32)")
    ap.add_argument("-o", "--out-file", type=str, default="out.jpg",
        help="the file to save the result (always overwrite)")
    ap.add_argument("-f", "--force-model", default=None,
        help="Force the use of the hog or cnn model", type=str)
    ap.add_argument("-w", "--waifu-executable", type=str, default=None,
        help="If set, it is used to upscale the images, results in smaller images actually being usable")

    args = vars(ap.parse_args())
    print(args)
    main(args)
