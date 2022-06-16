import cv2
import numpy as np
import onnxruntime as ort


def center_crop(image):
    # Crop center of frame
    height = image.shape[0]
    width = image.shape[1]
    edge = abs(height - width) // 2
    # Decide which way to crop
    if height > width:
        image = image[edge: edge + width]
    else:
        image = image[:, edge: edge + height]
    return image


def main():
    alphabet = list('ABCDEFGHIKLMNOPQRSTUVWXY')

    # Convert mean and std to 0-255 range
    mean = 0.485 * 255
    std = 0.229 * 255

    # executable session with onnx model
    onnx_sess = ort.InferenceSession("signlanguage.onnx")

    # initializing video capture object to link to camera feed
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()

        # preprocessing
        image = center_crop(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # resize to 28 x 28 for model
        x = cv2.resize(image, (28, 28))
        # normalize
        x = (x - mean) / std

        # reshape input and run model
        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = onnx_sess.run(None, {'input': x})[0]

        # Find index of largest value of output
        idx = np.argmax(y, axis=1)
        result = alphabet[int(idx)]

        # Display result on screen
        cv2.putText(image, result, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), thickness=3)
        cv2.imshow("Sign Language Translator", image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
