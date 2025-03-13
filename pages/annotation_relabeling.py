import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import glob
import platform
import plotly.express as px
from scripts.utils_pagebuttons import inject_custom_css, create_page_title, create_analysis_card
import colorsys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# --- Page Configuration ---
st.set_page_config(page_title="YOLO Annotation Relabeling", layout="wide")

# --- Inject shared CSS ---
inject_custom_css()

# --- Create page title with version info ---
version_info = f"Running on Python {'.'.join(platform.python_version_tuple())} | OpenCV {cv2.__version__} | PIL {Image.__version__}"
create_page_title("YOLO Annotation Relabeling Tool", version_info)

# Initialize class hierarchy in session state if not exists
if 'class_hierarchy' not in st.session_state:
    st.session_state.class_hierarchy = {}

# --- Class Management Interface ---
st.header("Class Management")

# Add new parent class
with st.expander("Add/Edit Classes", expanded=True):
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.subheader("Add Parent Class")
        new_parent_class = st.text_input("Enter new parent class name (e.g., Crab, Fish, etc.)")
        if st.button("Add Parent Class") and new_parent_class:
            if new_parent_class not in st.session_state.class_hierarchy:
                st.session_state.class_hierarchy[new_parent_class] = []
                st.success(f"Added parent class: {new_parent_class}")
            else:
                st.warning(f"Class {new_parent_class} already exists")
    
    with col2:
        st.subheader("Add Subclass")
        if st.session_state.class_hierarchy:
            parent_class = st.selectbox("Select parent class", list(st.session_state.class_hierarchy.keys()))
            new_subclass = st.text_input("Enter new subclass name (e.g., Red Rock Crab, Dungeness Crab)")
            if st.button("Add Subclass") and new_subclass and parent_class:
                if new_subclass not in st.session_state.class_hierarchy[parent_class]:
                    st.session_state.class_hierarchy[parent_class].append(new_subclass)
                    st.success(f"Added {new_subclass} under {parent_class}")
                else:
                    st.warning(f"Subclass {new_subclass} already exists under {parent_class}")
        else:
            st.info("Add a parent class first")

# Display current class hierarchy
if st.session_state.class_hierarchy:
    st.subheader("Current Class Hierarchy")
    for parent, subclasses in st.session_state.class_hierarchy.items():
        with st.expander(f"{parent} ({len(subclasses)} subclasses)"):
            for subclass in subclasses:
                col1, col2 = st.columns([3, 1])
                col1.write(f"- {subclass}")
                if col2.button("Remove", key=f"remove_{parent}_{subclass}"):
                    st.session_state.class_hierarchy[parent].remove(subclass)
            if st.button("Remove Parent Class", key=f"remove_parent_{parent}"):
                del st.session_state.class_hierarchy[parent]

    # Convert hierarchical classes to flat list for YOLO format
    flat_classes = []
    class_mapping = {}  # To store parent-child relationships
    
    for parent, subclasses in st.session_state.class_hierarchy.items():
        parent_idx = len(flat_classes)
        flat_classes.append(parent)
        class_mapping[parent] = {'index': parent_idx, 'subclasses': {}}
        
        for subclass in subclasses:
            subclass_idx = len(flat_classes)
            flat_classes.append(subclass)
            class_mapping[parent]['subclasses'][subclass] = subclass_idx

    # Update session state class names with flat list
    st.session_state.class_names = flat_classes
    st.session_state.class_mapping = class_mapping

# Save/Load Class Hierarchy
col1, col2 = st.columns(2)
with col1:
    if st.button("Export Class Hierarchy"):
        try:
            with open("class_hierarchy.json", "w") as f:
                json.dump(st.session_state.class_hierarchy, f, indent=4)
            st.success("Saved class hierarchy to class_hierarchy.json")
        except Exception as e:
            st.error(f"Error saving class hierarchy: {e}")

with col2:
    uploaded_file = st.file_uploader("Load Class Hierarchy", type=['json'])
    if uploaded_file is not None:
        try:
            st.session_state.class_hierarchy = json.load(uploaded_file)
            st.success("Loaded class hierarchy successfully")
        except Exception as e:
            st.error(f"Error loading class hierarchy: {e}")

st.markdown("---")

# --- Helper Functions ---
def generate_colors(num_classes):
    """Generate visually distinct colors for different classes"""
    colors = []
    for i in range(num_classes):
        # Use HSV color space for more visually distinct colors
        hue = i / num_classes
        saturation = 0.8 + (i % 3) * 0.1  # Slight variation in saturation
        value = 0.9
        
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Scale to 0-255 for OpenCV
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    # Add extra colors in case we have more classes than expected
    extras = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (128, 0, 0),    # Dark blue
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark red
    ]
    colors.extend(extras)
    
    return colors

def load_classes(classes_file=None, class_string=None):
    """Load class names from a file or a comma-separated string and initialize class hierarchy"""
    classes = []
    if classes_file is not None:
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f]
        except Exception as e:
            st.error(f"Error loading classes file: {e}")
            return []
    elif class_string is not None:
        classes = [cls.strip() for cls in class_string.split(',')]
    
    # Initialize class hierarchy with loaded classes as parents
    if classes and 'class_hierarchy' in st.session_state:
        # Only initialize if hierarchy is empty to avoid overwriting existing structure
        if not st.session_state.class_hierarchy:
            st.session_state.class_hierarchy = {cls: [] for cls in classes}
            st.info("Initialized class hierarchy with original classes as parents")
    
    return classes

def parse_yolo_annotation(annotation_path, image_width, image_height):
    """Parse YOLO format annotation file and convert to pixel coordinates"""
    annotations = []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # YOLO format: class_id x_center y_center width height
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * image_width
                    y_center = float(parts[2]) * image_height
                    width = float(parts[3]) * image_width
                    height = float(parts[4]) * image_height
                    
                    # Convert center coordinates to top-left for easier display
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': (x1, y1, x2, y2),
                        'original': {
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        }
                    })
    except Exception as e:
        st.warning(f"Could not load annotation: {e}")
    return annotations

def draw_bounding_boxes(image, annotations, class_names):
    """Draw bounding boxes on image with class labels, using different colors for each class"""
    img_with_boxes = image.copy()
    
    # Generate colors for classes
    colors = generate_colors(len(class_names) if class_names else 10)
    
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        class_id = ann['class_id']
        
        # Ensure class_id is within range
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
            color = colors[class_id % len(colors)]  # Use color based on class_id
        else:
            class_name = f"Unknown ({class_id})"
            color = (200, 200, 200)  # Gray for unknown classes
        
        # Draw rectangle with class-specific color
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Add class label with background
        text = class_name
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        
        # Draw background rectangle for text
        cv2.rectangle(img_with_boxes, 
                     (x1, y1 - text_h - 8), 
                     (x1 + text_w + 5, y1), 
                     color, 
                     -1)  # -1 fills the rectangle
        
        # Draw text (white on the colored background)
        cv2.putText(img_with_boxes, 
                   text, 
                   (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (255, 255, 255),  # White text
                   1)
    
    return img_with_boxes

def crop_roi(image, bbox):
    """Crop a region of interest from the image"""
    x1, y1, x2, y2 = bbox
    return image[max(0, y1):min(y2, image.shape[0]), max(0, x1):min(x2, image.shape[1])]

def save_yolo_annotation(annotations, output_path):
    """Save annotations back to YOLO format"""
    try:
        with open(output_path, 'w') as f:
            for ann in annotations:
                # Use original normalized coordinates
                original = ann['original']
                line = f"{ann['class_id']} {original['x_center']} {original['y_center']} {original['width']} {original['height']}\n"
                f.write(line)
        return True
    except Exception as e:
        st.error(f"Error saving annotation: {e}")
        return False

def save_classes_file(class_names, output_path):
    """Save class names to a classes.txt file"""
    try:
        with open(output_path, 'w') as f:
            for cls in class_names:
                f.write(f"{cls}\n")
        return True
    except Exception as e:
        st.error(f"Error saving classes file: {e}")
        return False

def extract_roi_features(roi_image):
    """Extract color, size, and shape features from ROI"""
    # Color features (average RGB)
    color_features = cv2.mean(roi_image)[:3]
    
    # Size features
    height, width = roi_image.shape[:2]
    size = width * height
    
    # Shape features
    aspect_ratio = width / height if height > 0 else 0
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    return {
        'color': color_features,
        'size': size,
        'shape': [aspect_ratio] + list(hu_moments)
    }

def cluster_rois(rois, method='color', n_clusters=3):
    """Cluster ROIs based on specified method"""
    features = []
    for roi_data in rois:
        roi_features = extract_roi_features(roi_data['roi'])
        if method == 'color':
            feature_vector = roi_features['color']
        elif method == 'size':
            feature_vector = [roi_features['size']]
        else:  # shape
            feature_vector = roi_features['shape']
        features.append(feature_vector)
    
    if not features:
        return []
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=min(n_clusters, len(features)))
    clusters = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to ROIs
    for roi_data, cluster in zip(rois, clusters):
        roi_data['cluster'] = int(cluster)
    
    return rois

# --- Main UI ---
st.markdown("""
### Introduction
This tool allows you to review and relabel object detections in YOLO annotation format. 
""")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data Loading")
    
    # Input for images directory
    images_dir = st.text_input("Images Directory (optional)", "")
    
    # Input for annotations directory
    annotations_dir = st.text_input("Annotations Directory (optional)", "")
    
    # File uploaders for images
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # File uploaders for annotations
    uploaded_annotations = st.file_uploader("Upload Annotations", type=["txt"], accept_multiple_files=True)
    
    # Class names input methods
    class_input_method = st.radio("Class Names Input Method", ["Upload classes.txt", "Enter class names"])
    
    if class_input_method == "Upload classes.txt":
        classes_file = st.file_uploader("Upload classes.txt", type=["txt"])
        class_names = []
        if classes_file is not None:
            class_names = load_classes(classes_file)
            if not st.session_state.class_hierarchy:
                st.success(f"Loaded {len(class_names)} parent classes from classes.txt")
    else:
        class_string = st.text_input("Enter class names (comma-separated)", "")
        class_names = load_classes(class_string=class_string) if class_string else []
        if class_names and not st.session_state.class_hierarchy:
            st.success(f"Initialized {len(class_names)} parent classes")
    
    # Show loaded classes
    if class_names:
        st.write("Loaded Classes:")
        # Show classes with their color
        colors = generate_colors(len(class_names))
        for i, cls in enumerate(class_names):
            color = colors[i % len(colors)]
            # Convert BGR to hex for displaying in HTML
            hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
            st.markdown(f"<div style='background-color:{hex_color};padding:2px 5px;color:white;border-radius:3px;'>{i}: {cls}</div>", 
                       unsafe_allow_html=True)
    
    # Load data button
    load_button = st.button("Load Data")

# Initialize session state for storing annotations data
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = []

if 'selected_roi' not in st.session_state:
    st.session_state.selected_roi = None

if 'filtered_class' not in st.session_state:
    st.session_state.filtered_class = None

if 'selected_rois' not in st.session_state:
    st.session_state.selected_rois = set()

if 'class_names' not in st.session_state:
    st.session_state.class_names = []

if 'show_image_explorer' not in st.session_state:
    st.session_state.show_image_explorer = False

if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

if 'selected_class_tab' not in st.session_state:
    st.session_state.selected_class_tab = None

if 'clustering_method' not in st.session_state:
    st.session_state.clustering_method = 'none'

# Update session state class names when they change
if class_names and (not st.session_state.class_names or class_names != st.session_state.class_names):
    st.session_state.class_names = class_names.copy()

# Load and process data when button is clicked
if load_button:
    st.session_state.loaded_data = []
    
    # Process directory inputs
    image_files = []
    if images_dir and os.path.isdir(images_dir):
        image_files.extend(glob.glob(os.path.join(images_dir, "*.jpg")))
        image_files.extend(glob.glob(os.path.join(images_dir, "*.jpeg")))
        image_files.extend(glob.glob(os.path.join(images_dir, "*.png")))
    
    # Process uploaded images
    uploaded_image_data = {}
    if uploaded_images:
        for uploaded_image in uploaded_images:
            # Convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            uploaded_image_data[uploaded_image.name] = {
                'image': image,
                'height': image.shape[0],
                'width': image.shape[1]
            }
    
    # Process annotations
    processed_data = []
    
    # Process directory annotations
    if annotations_dir and os.path.isdir(annotations_dir) and image_files:
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            image_basename = os.path.splitext(image_name)[0]
            annotation_path = os.path.join(annotations_dir, f"{image_basename}.txt")
            
            if os.path.exists(annotation_path):
                # Load the image for dimensions
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                height, width = image.shape[:2]
                
                # Parse annotations
                annotations = parse_yolo_annotation(annotation_path, width, height)
                
                processed_data.append({
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'image': image,
                    'annotations': annotations,
                    'width': width,
                    'height': height
                })
    
    # Process uploaded annotations
    if uploaded_annotations and uploaded_image_data:
        for uploaded_ann in uploaded_annotations:
            ann_name = uploaded_ann.name
            image_basename = os.path.splitext(ann_name)[0]
            
            # Try to find matching image
            for img_name, img_data in uploaded_image_data.items():
                if image_basename in img_name:
                    # Parse annotations
                    ann_content = uploaded_ann.read().decode('utf-8')
                    
                    # Write to temp file for parsing
                    temp_ann_path = f"temp_{ann_name}"
                    with open(temp_ann_path, 'w') as f:
                        f.write(ann_content)
                    
                    annotations = parse_yolo_annotation(
                        temp_ann_path, 
                        img_data['width'], 
                        img_data['height']
                    )
                    
                    # Remove temp file
                    if os.path.exists(temp_ann_path):
                        os.remove(temp_ann_path)
                    
                    processed_data.append({
                        'image_path': img_name,
                        'annotation_path': ann_name,
                        'image': img_data['image'],
                        'annotations': annotations,
                        'width': img_data['width'],
                        'height': img_data['height'],
                        'is_uploaded': True
                    })
                    break
    
    st.session_state.loaded_data = processed_data
    st.success(f"Loaded {len(processed_data)} images with annotations")

# Main content area
if st.session_state.loaded_data:
    # Image explorer toggle
    if st.button("Toggle Image Explorer"):
        st.session_state.show_image_explorer = not st.session_state.show_image_explorer
    
    # Paginated image explorer (when shown)
    if st.session_state.show_image_explorer:
        st.subheader("Image Explorer")
        
        # Pagination controls
        images_per_page = 6
        total_pages = (len(st.session_state.loaded_data) + images_per_page - 1) // images_per_page
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.session_state.current_page = st.slider("Page", 1, max(1, total_pages), st.session_state.current_page)
        
        # Display paginated images
        start_idx = (st.session_state.current_page - 1) * images_per_page
        end_idx = min(start_idx + images_per_page, len(st.session_state.loaded_data))
        
        grid_cols = st.columns(3)
        for i, item in enumerate(st.session_state.loaded_data[start_idx:end_idx]):
            col = grid_cols[i % 3]
            img_with_boxes = draw_bounding_boxes(item['image'], item['annotations'], st.session_state.class_names)
            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            col.image(img_with_boxes, caption=os.path.basename(item['image_path']), use_container_width=True)
    
    # ROI Management Section
    st.subheader("ROI Management")
    
    # Clustering controls
    col1, col2 = st.columns([3, 1])
    with col1:
        clustering_method = st.selectbox(
            "Cluster ROIs by",
            ['none', 'color', 'size', 'shape'],
            key='clustering_method'
        )
    
    # Extract all ROIs and organize by class
    all_rois_by_class = {}
    for img_idx, item in enumerate(st.session_state.loaded_data):
        for ann_idx, ann in enumerate(item['annotations']):
            class_id = ann['class_id']
            if class_id not in all_rois_by_class:
                all_rois_by_class[class_id] = []
            
            roi = crop_roi(item['image'], ann['bbox'])
            if roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                all_rois_by_class[class_id].append({
                    'roi': roi,
                    'roi_rgb': roi_rgb,
                    'img_idx': img_idx,
                    'ann_idx': ann_idx,
                    'class_id': class_id,
                    'image_name': os.path.basename(item['image_path'])
                })
    
    # Create tabs for each class
    if all_rois_by_class:
        tab_labels = []
        for class_id in sorted(all_rois_by_class.keys()):
            if 0 <= class_id < len(st.session_state.class_names):
                label = f"{st.session_state.class_names[class_id]} ({len(all_rois_by_class[class_id])})"
            else:
                label = f"Unknown {class_id} ({len(all_rois_by_class[class_id])})"
            tab_labels.append(label)
        
        tabs = st.tabs(tab_labels)
        
        # Display ROIs in each tab
        for idx, (class_id, rois) in enumerate(sorted(all_rois_by_class.items())):
            with tabs[idx]:
                if clustering_method != 'none':
                    rois = cluster_rois(rois, method=clustering_method)
                    # Group ROIs by cluster
                    clusters = {}
                    for roi in rois:
                        cluster = roi.get('cluster', 0)
                        if cluster not in clusters:
                            clusters[cluster] = []
                        clusters[cluster].append(roi)
                    
                    # Display ROIs by cluster
                    for cluster_id, cluster_rois in clusters.items():
                        st.write(f"Cluster {cluster_id + 1} ({len(cluster_rois)} ROIs)")
                        cols = st.columns(min(5, len(cluster_rois)))
                        for j, roi_data in enumerate(cluster_rois):
                            col = cols[j % len(cols)]
                            col.image(roi_data['roi_rgb'], use_container_width=True)
                            
                            # Delete button for each ROI
                            if col.button(f"Delete", key=f"delete_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"):
                                # Remove the annotation
                                st.session_state.loaded_data[roi_data['img_idx']]['annotations'].pop(roi_data['ann_idx'])
                            
                            # Relabel controls with hierarchical class selection
                            parent_class = col.selectbox(
                                "Parent Class",
                                list(st.session_state.class_hierarchy.keys()),
                                key=f"parent_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"
                            )
                            
                            subclasses = st.session_state.class_hierarchy[parent_class]
                            if subclasses:
                                new_class = col.selectbox(
                                    "Species",
                                    [f"{st.session_state.class_mapping[parent_class]['subclasses'][sub]}: {sub}" 
                                     for sub in subclasses],
                                    key=f"subclass_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"
                                )
                                new_class_id = int(new_class.split(":")[0])
                            else:
                                # If no subclasses, use parent class
                                new_class_id = st.session_state.class_mapping[parent_class]['index']
                            
                            if col.button("Apply", key=f"apply_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"):
                                st.session_state.loaded_data[roi_data['img_idx']]['annotations'][roi_data['ann_idx']]['class_id'] = new_class_id
                                
                            col.markdown("---")  # Visual separator between ROIs
                else:
                    # Display ROIs without clustering
                    cols = st.columns(5)
                    for j, roi_data in enumerate(rois):
                        col = cols[j % 5]
                        col.image(roi_data['roi_rgb'], use_container_width=True)
                        
                        # Delete button for each ROI
                        if col.button(f"Delete", key=f"delete_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"):
                            # Remove the annotation
                            st.session_state.loaded_data[roi_data['img_idx']]['annotations'].pop(roi_data['ann_idx'])
                        
                        # Relabel controls with hierarchical class selection
                        parent_class = col.selectbox(
                            "Parent Class",
                            list(st.session_state.class_hierarchy.keys()),
                            key=f"parent_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"
                        )
                        
                        subclasses = st.session_state.class_hierarchy[parent_class]
                        if subclasses:
                            new_class = col.selectbox(
                                "Species",
                                [f"{st.session_state.class_mapping[parent_class]['subclasses'][sub]}: {sub}" 
                                 for sub in subclasses],
                                key=f"subclass_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"
                            )
                            new_class_id = int(new_class.split(":")[0])
                        else:
                            # If no subclasses, use parent class
                            new_class_id = st.session_state.class_mapping[parent_class]['index']
                        
                        if col.button("Apply", key=f"apply_{class_id}_{roi_data['img_idx']}_{roi_data['ann_idx']}"):
                            st.session_state.loaded_data[roi_data['img_idx']]['annotations'][roi_data['ann_idx']]['class_id'] = new_class_id

    # Class management section
    st.subheader("Class Management")
    
    # Export classes
    export_col1, export_col2 = st.columns([3, 1])
    with export_col1:
        export_path = st.text_input("Path to save classes.txt file", "classes.txt")
    with export_col2:
        if st.button("Export Classes"):
            if save_classes_file(st.session_state.class_names, export_path):
                st.success(f"Saved {len(st.session_state.class_names)} classes to {export_path}")
    
    # View and edit classes
    with st.expander("View and Edit Classes"):
        # Display current classes in a table format
        if st.session_state.class_names:
            class_table = []
            for i, cls in enumerate(st.session_state.class_names):
                class_table.append({"ID": i, "Class Name": cls})
            
            st.write("Current Classes:")
            st.table(class_table)
    
    # Saving section
    st.subheader("Save Annotations")
    save_dir = st.text_input("Save Directory (leave empty to save in original location)", "")
    
    if st.button("Save All Annotations"):
        saved_count = 0
        for item in st.session_state.loaded_data:
            if 'is_uploaded' in item and item['is_uploaded']:
                # For uploaded files, save to the specified directory
                if save_dir:
                    output_path = os.path.join(save_dir, item['annotation_path'])
                else:
                    # If no save directory, save to current working directory
                    output_path = item['annotation_path']
            else:
                # For files loaded from disk
                if save_dir:
                    output_path = os.path.join(save_dir, os.path.basename(item['annotation_path']))
                else:
                    output_path = item['annotation_path']
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Save the annotation
            if save_yolo_annotation(item['annotations'], output_path):
                saved_count += 1
        
        st.success(f"Saved {saved_count} annotation files")

else:
    # Show instructions when no data is loaded
    st.info("Please use the sidebar to load your data")
    
    st.markdown("""
    ### How to use this tool:
    
    1. **Load your data**:
       - Either upload images and their corresponding annotation files (.txt)
       - Or provide directories containing your images and annotations
       - Provide class names through a classes.txt file or by entering them manually
    
    2. **Review and relabel**:
       - Browse through the loaded images and their annotations
       - Filter by class to focus on specific object types
       - Select ROIs to relabel them
       - Create new classes from selected ROIs
    
    3. **Save your changes**:
       - Export your updated class list to a classes.txt file
       - Choose a directory where to save the updated annotations
       - Click "Save All Annotations" to write your changes
    
    This tool maintains the YOLO format, so the output will be compatible with YOLO-based object detection applications.
    """)

    # Example annotation format
    st.subheader("YOLO Annotation Format")
    st.markdown("""
    ```
    class_id x_center y_center width height
    ```
    
    All values except class_id are normalized (0.0-1.0) to the image dimensions:
    - **class_id**: Integer class identifier (0, 1, 2, etc.)
    - **x_center, y_center**: Normalized coordinates of bbox center (0.0-1.0)
    - **width, height**: Normalized bbox dimensions (0.0-1.0)
    """)