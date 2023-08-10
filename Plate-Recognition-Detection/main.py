import cv2

width = 800
height = 400

# Load the number plate detector
n_plate_detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Create a directory to store detected license plate frames
output_directory = 'detected_frames'
os.makedirs(output_directory, exist_ok=True)

# Process 10 images
for i in range(1, 11):
    # Load the image, resize it, and convert it to grayscale
    image_path = f"{i}.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image {image_path}")
        continue

    image = cv2.resize(image, (width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the number plates in the grayscale image
    detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

    # Draw rectangles around the number plates and display the results
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(image, "Number plate detected", (x - 20, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

        # Extract the number plate from the grayscale image
        number_plate = gray[y:y + h, x:x + w]
        
        # Uncomment these *2* lines below to display cropped licence plate. note that we added 2 second delay to view the image.
        #cv2.imshow("Number plate", number_plate)
        #cv2.waitKey(2)
        
        # saves imaage when a number plate is detected.
        image_filename = os.path.join(output_directory, f"frame_{pbar.n:04d}.jpg")
        cv2.imwrite(image_filename, number_plate)

    cv2.imshow(f"Number plate detection - Image {i}", image)
    
    cv2.destroyAllWindows()
