import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_lottie import st_lottie
import requests

# Set page configuration
st.set_page_config(page_title="Advanced Dental Disease Detection", page_icon="🦷", layout="wide")

# Enhanced CSS for better styling and image sizing
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stAlert > div {
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .upload-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .condition-section {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-height: 400px;  /* Control maximum height of images */
        object-fit: contain;
    }
    .cropped-image {
        max-height: 250px;  /* Smaller height for cropped images */
        width: auto;
        margin: auto;
    }
    .st-tabs {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .detection-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    """
    Load Lottie animation from URL
    Args:
        url (str): URL of the Lottie animation
    Returns:
        dict: Lottie animation JSON data or None if failed to load
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie animation: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load the YOLO model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    """Process the image and return predictions"""
    try:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        results = model.predict(image_array)
        return results[0]
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def draw_single_condition(image, box, class_name):
    """Draw a single condition's bounding box on the image"""
    try:
        image_array = np.array(image).copy()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_array, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return Image.fromarray(image_array)
    except Exception as e:
        st.error(f"Error drawing single condition: {str(e)}")
        return image

def crop_detection(image, box):
    """Crop the region of the detected condition"""
    try:
        image_array = np.array(image)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        padding_x, padding_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
        height, width = image_array.shape[:2]
        x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
        x2, y2 = min(width, x2 + padding_x), min(height, y2 + padding_y)
        cropped = image_array[y1:y2, x1:x2]
        return Image.fromarray(cropped)
    except Exception as e:
        st.error(f"Error cropping detection: {str(e)}")
        return None

def draw_predictions(image, results):
    """Draw all bounding boxes and labels on the image"""
    try:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        plotted_image = results.plot()
        return Image.fromarray(plotted_image)
    except Exception as e:
        st.error(f"Error drawing predictions: {str(e)}")
        return image

def group_predictions_by_condition(results):
    """Group predictions by condition type"""
    condition_groups = {}
    if len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            if class_name not in condition_groups:
                condition_groups[class_name] = []
            condition_groups[class_name].append({'box': box, 'confidence': confidence})
    return condition_groups

def create_confidence_chart(condition_groups):
    data = []
    for condition, detections in condition_groups.items():
        for detection in detections:
            data.append({
                'Condition': condition,
                'Confidence': detection['confidence']
            })
    df = pd.DataFrame(data)
    fig = px.box(df, x='Condition', y='Confidence', points="all")
    fig.update_layout(title_text='Confidence Distribution by Condition')
    return fig

def create_condition_count_chart(condition_groups):
    counts = {condition: len(detections) for condition, detections in condition_groups.items()}
    fig = go.Figure(data=[go.Pie(labels=list(counts.keys()), values=list(counts.values()))])
    fig.update_layout(title_text='Distribution of Detected Conditions')
    return fig

def main():
    # Header
    st.title("🦷 Advanced Dental Disease Detection")
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.info(
            "This application uses YOLO11 to detect dental conditions in X-ray images. "
            "It can identify:\n"
            "- Cavities\n"
            "- Fillings\n"
            "- Impacted Teeth\n"
            "- Implants"
        )
        
        # Add Lottie animation
        #lottie_dental = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_xnbikipz.json")
        #if lottie_dental:
        #    st_lottie(lottie_dental, speed=1, height=200, key="dental")
    
    # Model loading
    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.error("Failed to load model. Please check the model path and try again.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                results = process_image(image, model)

            if results is not None:
                # Display original and processed images side by side
                st.header("Image Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("Detected Conditions")
                    processed_image = draw_predictions(image, results)
                    st.image(processed_image, use_container_width=True)

                # Group predictions by condition
                condition_groups = group_predictions_by_condition(results)

                if condition_groups:
                    st.header("Detailed Analysis by Condition")
                    
                    # Create tabs for each condition type
                    tabs = st.tabs(list(condition_groups.keys()))
                    
                    for tab, (condition_name, detections) in zip(tabs, condition_groups.items()):
                        with tab:
                            st.subheader(f"{condition_name} Detections")
                            st.write(f"Number of {condition_name} detected: {len(detections)}")
                            
                            # Display each instance of this condition
                            for idx, detection in enumerate(detections, 1):
                                st.write(f"#### Instance {idx}")
                                st.write(f"Confidence: {detection['confidence']:.2%}")
                                
                                # Create three columns with controlled image sizes
                                cols = st.columns(3)
                                
                                with cols[0]:
                                    st.write("Full Image with Detection")
                                    single_detection = draw_single_condition(image, detection['box'], condition_name)
                                    st.image(single_detection, use_container_width=True, clamp=True)
                                
                                with cols[1]:
                                    st.write("Cropped Region")
                                    cropped_region = crop_detection(image, detection['box'])
                                    if cropped_region is not None:
                                        st.image(cropped_region, use_container_width=True, clamp=True)
                                
                                with cols[2]:
                                    st.write("Cropped Region with Marking")
                                    if cropped_region is not None:
                                        marked_crop = draw_single_condition(cropped_region, detection['box'], condition_name)
                                        st.image(marked_crop, use_container_width=True, clamp=True)
                                
                                st.divider()
                    
                    # Add advanced visualizations
                    st.header("Advanced Visualizations")
                    viz_cols = st.columns(2)
                    
                    with viz_cols[0]:
                        confidence_chart = create_confidence_chart(condition_groups)
                        st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    with viz_cols[1]:
                        count_chart = create_condition_count_chart(condition_groups)
                        st.plotly_chart(count_chart, use_container_width=True)
                
                else:
                    st.info("No dental conditions detected in the image.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Additional information
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Upload a dental X-ray image using the file uploader above
        2. The model will automatically process the image
        3. Results will show detected conditions with confidence scores
        4. View detailed analysis for each condition type in separate tabs
        5. For each detection you'll see:
           - Full image with the detection marked
           - Cropped view of the detected region
           - Cropped view with detection marking
        6. Explore advanced visualizations for a comprehensive overview
        """)

    with st.expander("📊 Model Performance"):
        st.markdown("""
        Model metrics on validation dataset:
        - Overall mAP50: 0.603
        - Implant Detection (mAP50): 0.916
        - Filling Detection (mAP50): 0.827
        - Impacted Tooth Detection (mAP50): 0.644
        - Cavity Detection (mAP50): 0.0246
        """)

if __name__ == "__main__":
    main()