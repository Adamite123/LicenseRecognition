import os
import base64
from flask import Flask, Response, render_template, request, redirect, url_for, session, flash, jsonify,  after_this_request
import cv2
import numpy as np
import pytesseract
import imutils
import concurrent.futures
import mysql.connector
import re
import json 
from datetime import datetime
from werkzeug.utils import secure_filename
from base64 import b64decode
import time
import shutil
from pydub import AudioSegment
from pydub.playback import play
import threading
import pandas as pd
from openpyxl.utils import get_column_letter

app = Flask(__name__)
app.secret_key = '6800'  # set a secret key for the session


# Establish a connection to the MySQL database
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='fyp'
)

# Database connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'fyp',
}

# Create a cursor to interact with the database
cursor = db.cursor()

# Create a VideoCapture object to capture frames from the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 10)  # Set the desired FPS here (e.g., 5 FPS)

#=========== LIVE CAM ============

# Route to display the webcam stream and license numbers
@app.route('/camland')
def cam_landing():
    return render_template('camland.html')

# Route to fetch the latest license numbers from the database
@app.route('/license_numbers')
def get_license_numbers():
    filter_option = request.args.get('filter') # Get the filter option value from the request

    with mysql.connector.connect(**db_config) as db:
        with db.cursor() as cursor:
            #Log table
            cursor.execute("SELECT text, time FROM extracted_text ORDER BY id DESC")
            extracted_results = cursor.fetchall()
            extracted_license_numbers = [{'license_number': result[0], 'time': result[1].strftime('%Y-%m-%d %H:%M:%S')} for result in extracted_results]

            #saved table
            cursor.execute("SELECT license_number, time, status, image FROM saved_license_number ORDER BY id DESC")
            saved_results = cursor.fetchall()
            saved_license_numbers = [{'license_number': result[0], 'time': result[1].strftime('%Y-%m-%d %H:%M:%S'), 'status': result[2], 'image': result[3]} for result in saved_results]
            # print(saved_license_numbers)

            #activity table
            if filter_option == 'today':
                query = "SELECT license_number, time, desk FROM system_activity WHERE DATE(time) = CURDATE() ORDER BY id DESC"
            
            elif filter_option == 'yesterday':
                query = "SELECT license_number, time, desk FROM system_activity WHERE DATE(time) = CURDATE() - INTERVAL 1 DAY ORDER BY id DESC"
            
            else:
                query = "SELECT license_number, time, desk FROM system_activity ORDER BY id DESC"
            
            cursor.execute(query)
            activity_results = cursor.fetchall()
            activity_list = [{'license_number': result[0], 'time': result[1].strftime('%Y-%m-%d %H:%M:%S'), 'desk': result[2]} for result in activity_results]

            return json.dumps({'license_numbers': extracted_license_numbers, 'saved_license_numbers': saved_license_numbers, 'activity_list': activity_list})

# Function to play the beep sound in a separate thread
def play_beep_sound():
    beep = AudioSegment.from_file("static/beep.mp3")  # Replace "beep.wav" with the path to your beep sound file
    play(beep)

# Function to preprocess the captured frame
def preprocess_frame(frame):
    # Enable OpenCL support in OpenCV
    # cv2.ocl.setUseOpenCL(True)

    img = imutils.resize(frame, width=800)

    # Convert the image to UMat for GPU acceleration
    # img = cv2.UMat(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Convert UMat back to numpy array
    thresh_np = np.array(thresh)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    license_plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area > 700 and perimeter > 150:
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                license_plates.append((x, y, w, h))

    # Perform Non-Maximum Suppression (NMS) to eliminate overlapping license plates
    license_plates = np.array(license_plates)
    scores = np.zeros(len(license_plates))

    for i in range(len(license_plates)):
        x1, y1, w1, h1 = license_plates[i]
        x2 = x1 + w1
        y2 = y1 + h1

        for j in range(i + 1, len(license_plates)):
            x3, y3, w3, h3 = license_plates[j]
            x4 = x3 + w3
            y4 = y3 + h3

            # Calculate the intersection area
            x5 = max(x1, x3)
            y5 = max(y1, y3)
            x6 = min(x2, x4)
            y6 = min(y2, y4)
            intersection_area = max(0, x6 - x5) * max(0, y6 - y5)

            # Calculate the union area
            area1 = w1 * h1
            area2 = w3 * h3
            union_area = area1 + area2 - intersection_area

            # Calculate the intersection over union (IoU)
            iou = intersection_area / union_area

            # If the IoU is above a threshold, suppress the license plate with the lower score
            if iou > 0.5:
                scores[i] = -1 if scores[i] < scores[j] else -1
                scores[j] = -1 if scores[j] < scores[i] else -1

    filtered_license_plates = []
    for i in range(len(license_plates)):
        if scores[i] != -1:
            filtered_license_plates.append(license_plates[i])

    # Process each license plate
    def process_license_plate(plate):
        x, y, w, h = plate

        # Convert UMat back to numpy array
        # img_np = img.get()
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

        return char_text, plate

    # Process each license plate in parallel using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_license_plate, filtered_license_plates)

    recognized_text = []  # List to store recognized text

    # Check OCR results and draw rectangles around license plates
    for result in results:

        if result is not None:
            char_text, (x, y, w, h) = result

            if (
                re.match(r'^[A-Za-z0-9]{7}$', char_text) and
                any(char.isalpha() for char in char_text) and
                any(char.isdigit() for char in char_text)
                ):

                #remove special characters
                char_text = char_text.replace(" ", "")
                char_text = re.sub('[^A-Za-z0-9]+', '', char_text)

                # Establish a new database connection for each iteration
                with mysql.connector.connect(**db_config) as db:
                    with db.cursor() as cursor:
                    
                        # Check if the license number exists in the saved_license_number table with status 'blocked'
                        cursor.execute("SELECT status FROM saved_license_number WHERE license_number = %s", (char_text,))
                        blocked_result = cursor.fetchall()
                        
                        if blocked_result:
                            # print(blocked_result[0][0])
                            if blocked_result[0][0] == 'blocked':
                                
                                # Draw a rectangle around the license plate
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                
                                # Draw the recognized text on the image
                                cv2.putText(img, "BLOCKED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                                threading.Thread(target=activity, args=('scanned_block', char_text)).start()

                                # Play the beep sound in a separate thread
                                threading.Thread(target=play_beep_sound).start()

                            else :
                                # Draw a rectangle around the license plate
                                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                
                                # Draw the recognized text on the image
                                cv2.putText(img, "REGISTERED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                                threading.Thread(target=activity, args=('scanned_blue', char_text)).start()
                        
                        else :
                            # Draw a rectangle around the license plate
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                            # Draw the recognized text on the image
                            cv2.putText(img, char_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                        # Consume the result set before executing the next query
                        cursor.nextset()
                    
                        # Append the recognized text to the list
                        recognized_text.append(char_text)
    

    # Join the recognized text into a single string
    recognized_text_str = ', '.join(recognized_text)

    return img, recognized_text_str


# Route to display the webcam stream
@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            # Establish a new database connection for each iteration
            with mysql.connector.connect(**db_config) as db:
                with db.cursor() as cursor:
                    ret, frame = cap.read()
                    processed_frame, recognized_text = preprocess_frame(frame)
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    jpeg = buffer.tobytes()

                    # Insert the recognized text into the database
                    if recognized_text:
                        license_numbers = recognized_text.split(', ')
                        
                        for license_number in license_numbers:
                            license_number = license_number.strip().upper()
                            
                            # Check if the license number meets the classification criteria
                            if (
                                re.match(r'^[A-Za-z0-9]{7}$', license_number) and
                                any(char.isalpha() for char in license_number) and
                                any(char.isdigit() for char in license_number)
                            ):
                                # Check if the license number already exists in the extracted_text table
                                cursor.execute("SELECT * FROM extracted_text WHERE text = %s", (license_number,))
                                extracted_result = cursor.fetchone()

                                # Consume the result set before executing the next query
                                cursor.nextset()

                                # Check if the license number already exists in the saved_license_number table
                                cursor.execute("SELECT * FROM saved_license_number WHERE license_number = %s", (license_number,))
                                saved_result = cursor.fetchone()

                                # Consume the result set before executing the next query
                                cursor.nextset()

                                if not extracted_result and not saved_result:
                                    # License number doesn't exist in either table, perform the insertion
                                    # Generate a unique filename based on the current timestamp
                                    filename = f"{recognized_text}_{int(time.time())}.jpg"

                                    # Save the license plate image as a file
                                    filepath = os.path.join("static/log_pict", filename)
                                    cv2.imwrite(filepath, processed_frame)

                                    # Insert the filename and recognized text into the database
                                    sql = "INSERT INTO extracted_text (text, image) VALUES (%s, %s)"
                                    val = (recognized_text, filename)
                                    cursor.execute(sql, val)
                                    db.commit()

                    # Create the multipart response
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                           b'Content-Type: text/plain\r\n\r\n' + recognized_text.encode() + b'\r\n\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_license/<license_number>')
def save_license(license_number):
    # Search for the license plate in the extracted_text table
    cursor.execute("SELECT * FROM extracted_text WHERE text = %s", (license_number,))
    result = cursor.fetchone()

    if result:
        # License plate found in extracted_text table
        # Move the license plate image to the static/images directory
        old_filepath = os.path.join("static/log_pict", result[3])
        new_filename = f"{result[1]}_{int(time.time())}.jpg"
        new_filepath = os.path.join("static/images", new_filename)
        shutil.move(old_filepath, new_filepath)

        # Move the license plate to saved_license_number table
        cursor.execute("INSERT INTO saved_license_number (license_number, time, status, image) VALUES (%s, %s, 'registered', %s)",
                       (result[1], result[2], new_filename))

        # Remove the license plate from extracted_text table
        cursor.execute("DELETE FROM extracted_text WHERE text = %s", (license_number,))
        db.commit()

        activity('saved',license_number)

        flash('License Number Saved!', 'error')
        return redirect('/camland')
    else:
        # License plate not found in extracted_text table
        flash('License Number Failed to Save!', 'error')
        return redirect('/camland')


# Remove all License Number
@app.route('/removeall')
def removeall():

    # Retrieve all filenames from the `extracted_text` table
    cursor.execute("SELECT image FROM extracted_text")
    rows = cursor.fetchall()
    filenames = [row[0] for row in rows]

    # Delete the rows from the `extracted_text` table
    cursor.execute("DELETE FROM `extracted_text`")
    db.commit()

    # Delete the image files
    for filename in filenames:
        filepath = os.path.join("static/log_pict", filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    activity('dell_all','none')

    flash('License Number Log Removed!', 'log')
    return redirect('/camland')


# Block License Number
@app.route('/block/<license_number>')
def block(license_number):
    
    cursor.execute("UPDATE saved_license_number SET status = 'blocked' WHERE license_number = %s", (license_number,))
    db.commit()

    activity('block', license_number)

    flash('License Number Blocked!', 'error')
    return redirect('/camland')


# Unblock License Number
@app.route('/unblock/<license_number>')
def unblock(license_number):
    
    cursor.execute("UPDATE saved_license_number SET status = 'registered' WHERE license_number = %s", (license_number,))
    db.commit()

    activity('unblock', license_number)

    flash('License Number Unblocked!', 'error')
    return redirect('/camland')

def activity(trigger, license_number):
    
    if trigger == 'scanned_block':
        cursor.execute("INSERT INTO `system_activity`(`license_number`, `desk`) VALUES (%s, %s)", (license_number,"Blocked license plate scanned by camera"))
        db.commit()
    
    elif trigger == 'scanned_blue':
        cursor.execute("INSERT INTO `system_activity`(`license_number`, `desk`) VALUES (%s, %s)", (license_number,"Registered license plate scanned by camera"))
        db.commit()
    
    elif trigger == 'saved':
        cursor.execute("INSERT INTO `system_activity`(`license_number`, `desk`) VALUES (%s, %s)", (license_number,"User Saved License Plate"))
        db.commit()

    elif trigger == 'block':
        cursor.execute("INSERT INTO `system_activity`(`license_number`, `desk`) VALUES (%s, %s)", (license_number,"User Block License Plate"))
        db.commit()

    elif trigger == 'unblock':
        cursor.execute("INSERT INTO `system_activity`(`license_number`, `desk`) VALUES (%s, %s)", (license_number,"User Unblock License Plate"))
        db.commit()
    
    elif trigger == 'dell_all':
        cursor.execute("INSERT INTO `system_activity`(`license_number`, `desk`) VALUES (%s, %s)", ("none","Deleted All Scanned LOG"))
        db.commit()
    
    else:
        print(trigger)

#=========== Excel ===========
@app.route('/download_excel', methods=['GET'])
def download_data():
    # Get the current date
    current_date = datetime.now().date()

    # Execute the MySQL query to fetch the data for the current year, month, and date
    query = "SELECT * FROM system_activity WHERE DATE(time) = '{}'".format(current_date.strftime('%Y-%m-%d'))
    df = pd.read_sql_query(query, db)

    # Convert the DataFrame to an Excel file
    file_name = 'activity_report_{}.xlsx'.format(current_date.strftime('%Y-%m-%d'))
    excel_file = pd.ExcelWriter(file_name, engine='openpyxl')
    df.to_excel(excel_file, index=False)
    excel_file.book.save(file_name)  # Save the Excel file
    excel_file.close()

    # Create a response with the Excel file
    response = Response(
        open(file_name, 'rb'),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response.headers.set("Content-Disposition", "attachment", filename=file_name)

    return response


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

@app.route('/image_upload')
def image_upload():
    # Retrieve the license entries from the database
    with mysql.connector.connect(**db_config) as db:
        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM saved_license_number")
            rows = cursor.fetchall()

            # Define a class to represent a license entry
            class LicenseEntry:
                def __init__(self, id, license_number, time, status, image_url):
                    self.id = id
                    self.license_number = license_number
                    self.time = time
                    self.status = status
                    self.image_url = image_url

            # Create LicenseEntry objects for each row in the database
            license_entries = []
            for row in rows:
                entry = LicenseEntry(row[0], row[1], row[2], row[3], row[4])
                license_entries.append(entry)

    # Render the image_upload.html template with the license entries
    return render_template('image_upload.html', license_entries=license_entries)


# Image preprocessing
@app.route('/images', methods=['GET', 'POST'])
def upload_file():
    ocr = []
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

                #remove special characters
                char_text = char_text.replace(" ", "")
                char_text = re.sub('[^A-Za-z0-9]+', '', char_text)

                # Check if OCR result is a mix of letters and numbers with a length of 4
                if len(char_text) >= 4 and any(char.isalpha() for char in char_text) and any(char.isdigit() for char in char_text):
                    # Establish a new database connection for each iteration
                    with mysql.connector.connect(**db_config) as db:
                        with db.cursor() as cursor:
                        
                            # Check if the license number exists in the saved_license_number table with status 'blocked'
                            cursor.execute("SELECT * FROM saved_license_number WHERE license_number = %s AND status = 'blocked'", (char_text,))
                            blocked_result = cursor.fetchone()

                            if (blocked_result):
                                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                cv2.putText(img, "BLOCKED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            else:
                                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    ocr.append(char_text)

                    # Encode the image to JPEG format for display
                    ret, jpeg = cv2.imencode('.jpg', img)
                    frame = jpeg.tobytes()
            
            # Return the processed image
            return render_template('image.html', frame=frame, ocr=ocr)

        else:
            return "Invalid file format. Only JPEG and PNG images are allowed."
    
    # If the request method is not POST, return the upload form
    return redirect(url_for('image_upload'))

@app.route('/save_license_image', methods=['POST'])
def save_license_image():
    
    image_data = request.form['image_data']
    license_number = request.form['license_number']

    # Convert the base64 image data to bytes
    image_bytes = base64.b64decode(image_data)

    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.jpg"

    # Save the image to the 'images' folder
    images_folder = 'static/images'
    os.makedirs(images_folder, exist_ok=True)
    image_path = os.path.join(images_folder, filename)
    with open(image_path, 'wb') as image_file:
        image_file.write(image_bytes)

    # Save the filename and other information in the database
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO saved_license_number (license_number, time, status, image) VALUES (%s, %s, 'registered', %s)",
        (license_number, current_time, filename)
    )
    db.commit()

    # Flash a success message
    flash(f'License Number "{license_number}" Saved', 'success')

    return redirect(url_for('image_upload'))


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
            flash('Invalid login', 'error')
            return redirect('/login')

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