import cv2
import easyocr
import numpy as npq
import os

def preprocess_image(image):
    """Preprocess the image to make number plate characters more recognizable."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return cleaned

def detect_number_plate_region(frame):
    """Detect and isolate the number plate region using contours."""
    preprocessed_image = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    number_plate_candidates = []
    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check for rectangular contours
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h

            # Filter based on aspect ratio
            if 2 < aspect_ratio < 5:  # Typical number plate aspect ratio
                number_plate_candidates.append((x, y, w, h))

    # Sort by area (largest first)
    number_plate_candidates = sorted(number_plate_candidates, key=lambda b: b[2] * b[3], reverse=True)
    return number_plate_candidates[:1]  # Return the top candidate

def detect_and_read_number_plate(frame):
    """Detect and read the number plate from an image frame."""
    # Detect number plate region
    candidates = detect_number_plate_region(frame)

    detected_texts = []
    for (x, y, w, h) in candidates:
        # Extract the ROI
        roi = frame[y:y+h, x:x+w]

        # Resize the ROI for better OCR
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Convert ROI to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Use EasyOCR to detect and read text
        reader = easyocr.Reader(['en'])
        results = reader.readtext(roi_gray)

        # Draw bounding boxes and text on the original image
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Confidence threshold
                detected_texts.append(text)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, detected_texts

if __name__ == "__main__":
    # Path to the single image
    image_path = "pl1.jpeg"

    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}.")
        exit()

    # Detect and read the number plate
    annotated_frame, detected_texts = detect_and_read_number_plate(frame)

    # Display the result
    print(f"Detected Texts: {detected_texts}")
    cv2.imshow("Number Plate Recognition", annotated_frame)

    # Wait for key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

