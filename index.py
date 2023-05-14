import os
import base64
from flask import Flask, Response, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import pytesseract
import imutils
import concurrent.futures

app = Flask(__name__)
app.secret_key = '6800'  # set a secret key for the session


# Create a VideoCapture object to capture frames from the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 3)  # Set the desired FPS here (e.g., 5 FPS)

#=========== LIVE CAM ============

@app.route('/tes')
def tes():
    return render_template('tes.html')

extracted_text = ""

@app.route('/extracted_text')
def get_extracted_text():
    global extracted_text
    return extracted_text


# Create a set to store processed license plates
processed_plates = set()

# Function to preprocess the captured frame
def preprocess_frame(frame):
    global processed_plates

    img = imutils.resize(frame, width=800)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and filter out the ones that are not license plates
    license_plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area > 100 and perimeter > 100:
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:
                NumberPlateCnt = approx
                x, y, w, h = cv2.boundingRect(cnt)
                license_plates.append((x, y, w, h))

    # Process each license plate
    def process_license_plate(plate):
        x, y, w, h = plate

        # Crop the license plate from the image
        plate_img = img[max(0, y + 2):min(img.shape[0], y + h - 2), max(0, x + 2):min(img.shape[1], x + w - 2)]

        # Convert the cropped license plate to grayscale
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        plate_blur = cv2.bilateralFilter(plate_gray, 20, 40, 75)

        # Apply dilation to slightly increase character thickness
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilated = cv2.dilate(plate_blur, kernel)

        # Apply adaptive thresholding to binarize the image
        plate_thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Perform OCR on the processed image
        char_text = pytesseract.image_to_string(plate_thresh, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

        if len(char_text) >= 4 and any(char.isalpha() for char in char_text) and any(char.isdigit() for char in char_text):
            # Check if the license plate has already been processed
            if char_text not in processed_plates:
                processed_plates.add(char_text)
                return char_text, plate

    # Process each license plate in parallel using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_license_plate, license_plates)

    recognized_text = []  # List to store recognized text

    # Check OCR results and draw rectangles around license plates
    for result in results:
        if result is not None:
            char_text, (x, y, w, h) = result

            # Draw a rectangle around the license plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw the recognized text on the image
            cv2.putText(img, char_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Append the recognized text to the list
            recognized_text.append(char_text)

    # Convert recognized_text list to a string
    recognized_text_str = ', '.join(recognized_text)

    # Update the extracted_text variable
    global extracted_text
    extracted_text = recognized_text_str

    return img

# Route to display the webcam stream
@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            ret, frame = cap.read()
            processed_frame = preprocess_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            jpeg = buffer.tobytes()

            # Fetch the extracted text
            extracted_text = get_extracted_text()

            # Create the multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + extracted_text.encode() + b'\r\n\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')



#=========== IMAGE ===========

# Set the allowed image extensions
ALLOWED_EXTENSIONS_IMAGES = {'jpg', 'jpeg', 'png'}

def allowed_file_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGES

# Define the b64encode filter
@app.template_filter('b64encode')
def b64encode_filter(s):
    if isinstance(s, str):
        s = s.encode('utf-8')
    return base64.b64encode(s).decode('utf-8')

@app.route('/')
def index():
    if 'logged_in' in session and session['logged_in']:
        return render_template('index.html', username=session['username'])
    else:
        return redirect('/login')
    # return render_template('index.html')

# Image preprocessing
@app.route('/images', methods=['GET', 'POST'])
def upload_file():
    ocr = ''
    if request.method == 'POST':
        # Get the uploaded file
        file_images = request.files['image']
        
        # Check if the file is allowed
        if file_images and allowed_file_image(file_images.filename):
            # Read the image as a NumPy array
            img_np = np.fromfile(file_images, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            img = imutils.resize(img, width=800)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            # cv2.imshow(gray_blur)

            # Apply adaptive thresholding to the grayscale image
            thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # cv2.imshow(thresh)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate over the contours and filter out the ones that are not license plates
            license_plates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)

                if area > 100 and perimeter > 100:
                    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                    if len(approx) == 4:
                        NumberPlateCnt = approx
                        x, y, w, h = cv2.boundingRect(cnt)
                        license_plates.append((x, y, w, h))

            # Process each license plate
            for (x, y, w, h) in license_plates:

                # Crop the license plate from the image
                plate_img = img[max(0, y + 2):min(img.shape[0], y + h - 2), max(0, x + 2):min(img.shape[1], x + w - 2)]

                # Convert the cropped license plate to grayscale
                plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

                # Apply blur to reduce noise
                plate_blur = cv2.bilateralFilter(plate_gray, 20, 40, 75)
                # Apply dilation to slightly increase character thickness
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
                dilated = cv2.dilate(plate_blur, kernel)

                # Apply adaptive thresholding to binarize the image
                plate_thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                # print("== Thresh ==")
                # cv2.imshow(dilated)

                # Perform OCR on the processed image
                char_text = pytesseract.image_to_string(plate_thresh, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
                # print("Result1 = ",char_text)

                # Check if OCR result is a mix of letters and numbers with a length of 4
                if len(char_text) >= 4 and any(char.isalpha() for char in char_text) and any(char.isdigit() for char in char_text):
                    # Draw a rectangle around the license plate
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                char_text = pytesseract.image_to_string(plate_thresh, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
                
                if len(char_text) >= 4 and any(char.isalpha() for char in char_text) and any(char.isdigit() for char in char_text):
                    ocr += char_text
                    print("Result = ",char_text)

            # Encode the image to JPEG format for display
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            
            # Return the processed image
            return render_template('image.html', frame=frame, ocr=ocr)

        else:
            return "Invalid file format. Only JPEG and PNG images are allowed."
    
    # If the request method is not POST, return the upload form
    return render_template('image_upload.html')


#=========== VIDEO ==========

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/video')
def video():
    return render_template('upload_vids.html')

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        # image processing on each frame
        img = imutils.resize(frame, width=800)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # cv2.imshow(gray_blur)

        # Apply adaptive thresholding to the grayscale image
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # cv2.imshow(thresh)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the contours and filter out the ones that are not license plates
        license_plates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if area > 1000 and perimeter > 100:
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                if len(approx) == 4:
                    NumberPlateCnt = approx
                    x, y, w, h = cv2.boundingRect(cnt)
                    license_plates.append((x, y, w, h))
        
        # Define the color for the bounding rectangles
        color = (0, 255, 0)

        # Draw bounding rectangles around license plates and show the result
        for plate in license_plates:
            x, y, w, h = plate
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # yield the processed frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img)[1].tobytes() + b'\r\n')

    cap.release()

@app.route('/video/upload', methods=['GET', 'POST'])
def upload_video():
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # process each frame of the uploaded video using OpenCV
        return Response(process_video(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploads/<filename>')
def display_video(filename):
    return render_template('upload_vids.html', filename=filename)


# ========== LOGIN ROUTE ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # check if the credentials are valid
        if username == 'admin' and password == '123':
            # store the user's information in the session
            session['username'] = username
            session['logged_in'] = True
            return redirect('/')
        else:
            return 'Invalid login credentials'

    # if the request method is GET, show the login form
    return render_template('login.html')

@app.route('/logout')
def logout():
    # clear the session data
    session.clear()
    # redirect the user to the login page
    return redirect('/login')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)