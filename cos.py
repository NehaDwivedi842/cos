import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_camera import Camera
import numpy as np
import imutils
from skimage import io
import math
from ultralytics import YOLO

app = dash.Dash(__name__)

# Function to detect objects, annotate image, and calculate tonnage
def detect_objects_and_annotate(image, num_cavities, tons_per_inch_sq):
    model = YOLO("yolov8m-seg-custom.pt")
    results = model.predict(source=image, show=False)
    pixel_per_cm = None
    if results:
        pixel_per_cm = None
        for result in results:
            bounding_boxes = result.boxes.xyxy
            for box in bounding_boxes:
                x1, y1, x2, y2 = box[:4].int().tolist()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                radius = max(abs(x2 - x1), abs(y2 - y1)) // 2
                cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
                ref_w, ref_h = abs(x2 - x1), abs(y2 - y1)
                dist_in_pixel = max(ref_w, ref_h)
                ref_coin_diameter_cm = 2.426
                pixel_per_cm = dist_in_pixel / ref_coin_diameter_cm
                ref_text = " Size=0.955"
                cv2.putText(image, ref_text, (center_x - 150, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
        if pixel_per_cm is None:
            st.error("No Reference object detected in the image. Please recapture.")
            return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    filtered_contours = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > 50:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            contour_in_yolo_object = False
            for yolo_box in bounding_boxes:
                yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box[:4].int().tolist()
                if yolo_x1 < rect[0][0] < yolo_x2 and yolo_y1 < rect[0][1] < yolo_y2:
                    contour_in_yolo_object = True
                    break
            if not contour_in_yolo_object:
                filtered_contours.append(cnt)

    largest_contour = max(filtered_contours, key=cv2.contourArea)
    if largest_contour is not None:
        cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)
        area_cm2 = cv2.contourArea(largest_contour) / (pixel_per_cm ** 2)
        rect = cv2.minAreaRect(largest_contour)
        (x, y), (width_px, height_px), angle = rect
        width_cm = width_px / pixel_per_cm
        height_cm = height_px / pixel_per_cm
        if area_cm2 < 1:
            aspect_ratio = width_px / height_px
            if aspect_ratio < 1:
                perimeter = cv2.arcLength(largest_contour, True)
                diameter = perimeter / math.pi
                area_cm2 = (diameter / 2) ** 2 * math.pi
        text_x = int(x - 100)
        text_y = int(y - 20)
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        area_in2 = area_cm2 / 2.54
        tonnage = calculate_tonnage(area_in2, num_cavities, tons_per_inch_sq)
        cv2.putText(image, "Length: {:.1f}in".format(width_in), (text_x, text_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, "Breadth: {:.1f}in".format(height_in), (text_x, text_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, "Area: {:.1f}in^2".format(area_in2), (text_x, text_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, "Tonnage: {:.2f}".format(tonnage), (text_x, text_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(image, "Coin is the reference Object", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 1), 2)
    return image

# Function to calculate tonnage
def calculate_tonnage(area_in2, num_cavities, tons_per_inch_sq):
    tonnage = area_in2 * num_cavities * tons_per_inch_sq
    return tonnage

app.layout = html.Div([
    html.H1("Object Detection and Size Estimation"),
    html.Div([
        html.Div([
            html.Button("Capture Image and Calculate Tonnage", id="capture-button"),
            dcc.Loading(id="loading-icon", type="default", children=[html.Div(id="captured-image")]),
            dcc.Store(id="captured-image-store"),
            html.Button("Process Image", id="process-button"),
            html.Div(id="image-annotation"),
            html.Div(id="image-dimensions")
        ], style={'textAlign': 'center'}),
        html.Div([
            html.H3("Camera Preview"),
            Camera(id="live-camera")
        ], style={'textAlign': 'center'})
    ])
])

@app.callback(
    Output("captured-image-store", "data"),
    [Input("capture-button", "n_clicks")]
)
def capture_image(n_clicks):
    if n_clicks:
        frame = Camera().take_snapshot()
        frame_encoded = cv2.imencode('.jpg', frame)[1].tostring()
        return frame_encoded

@app.callback(
    Output("captured-image", "children"),
    [Input("captured-image-store", "data")]
)
def display_captured_image(frame_encoded):
    if frame_encoded:
        return html.Img(src='data:image/jpg;base64,{}'.format(frame_encoded.decode()))

@app.callback(
    [Output("image-annotation", "children"),
     Output("image-dimensions", "children")],
    [Input("process-button", "n_clicks")],
    [State("captured-image-store", "data")]
)
def process_image(n_clicks, frame_encoded):
    if n_clicks and frame_encoded:
        nparr = np.frombuffer(frame_encoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = frame.copy()
        num_cavities = 1  # You can set the default values or use input fields to get user input
        tons_per_inch_sq = 1.0  # You can set the default values or use input fields to get user input
        annotated_image = detect_objects_and_annotate(image, num_cavities, tons_per_inch_sq)
        if annotated_image is not None:
            _, encoded_image = cv2.imencode('.jpg', annotated_image)
            encoded_image_str = encoded_image.tostring()
            annotated_image_src = 'data:image/jpg;base64,{}'.format(encoded_image_str.decode())
            return html.Img(src=annotated_image_src), html.Div("Dimensions and tonnage calculated successfully.")
        else:
            return html.Div("Error processing image."), None
    else:
        return None, None

if __name__ == "__main__":
    app.run_server(debug=True)
