import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📸",
    layout="wide"
)

# Create necessary directories
Path("data").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)

ATTENDANCE_FILE = "data/attendance.csv"
ATTENDANCE_EXCEL_FILE = "data/attendance.xlsx"
USERS_FILE = "data/users.csv"

# Initialize session state
if 'users_df' not in st.session_state:
    if os.path.exists(USERS_FILE):
        st.session_state.users_df = pd.read_csv(USERS_FILE)
    else:
        st.session_state.users_df = pd.DataFrame(columns=['ID', 'Name', 'Photo_Path', 'RegisterDate'])

if 'temp_image' not in st.session_state:
    st.session_state.temp_image = None

if 'registration_success' not in st.session_state:
    st.session_state.registration_success = False

if 'detected_faces' not in st.session_state:
    st.session_state.detected_faces = []

# Load YOLOv8 face detection model
@st.cache_resource
def load_face_detector():
    """
    Load YOLOv8 model for face detection
    Using YOLOv8n (nano) for faster inference
    """
    try:
        # Try to load a pre-trained YOLOv8 model
        # You can use 'yolov8n.pt' for general object detection
        # or a face-specific model if available
        model = YOLO('yolov8n-face.pt')  # Use face-specific model if available
    except:
        # Fallback to standard YOLOv8n and filter for person class
        model = YOLO('yolov8n.pt')
    return model

yolo_model = load_face_detector()

# Save users database
def save_users():
    st.session_state.users_df.to_csv(USERS_FILE, index=False)

# Load attendance records
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)
    else:
        return pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time', 'Status'])

# Save attendance
def save_attendance(df):
    # Save as CSV
    df.to_csv(ATTENDANCE_FILE, index=False)
    
    # Save as Excel with formatting
    try:
        # Create Excel writer object
        with pd.ExcelWriter(ATTENDANCE_EXCEL_FILE, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Attendance']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4CAF50',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            cell_format = workbook.add_format({
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            # Write headers with formatting
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet.set_column('A:A', 15)  # ID
            worksheet.set_column('B:B', 25)  # Name
            worksheet.set_column('C:C', 15)  # Date
            worksheet.set_column('D:D', 12)  # Time
            worksheet.set_column('E:E', 12)  # Status
            
            # Apply cell formatting to data
            for row_num in range(1, len(df) + 1):
                for col_num in range(len(df.columns)):
                    worksheet.write(row_num, col_num, df.iloc[row_num-1, col_num], cell_format)
    except Exception as e:
        st.warning(f"Excel file creation failed: {e}. CSV file saved successfully.")

# Mark attendance
def mark_attendance(user_id, name):
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Check if already marked today
    if not df.empty:
        existing = df[(df['ID'] == user_id) & (df['Date'] == today)]
        if not existing.empty:
            return False, "Already marked today"
    
    # Add new record
    new_record = pd.DataFrame([{
        'ID': user_id,
        'Name': name,
        'Date': today,
        'Time': datetime.now().strftime("%H:%M:%S"),
        'Status': 'Present'
    }])
    
    df = pd.concat([df, new_record], ignore_index=True)
    save_attendance(df)
    return True, "Attendance marked successfully"

# Detect faces using YOLOv8
def detect_faces(image):
    """
    Detect faces using YOLOv8 model
    Returns faces in format compatible with OpenCV (x, y, w, h)
    """
    try:
        # Run YOLOv8 inference
        results = yolo_model(image, conf=0.4, verbose=False)
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to OpenCV format (x, y, width, height)
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # Filter for reasonable face sizes
                if w > 30 and h > 30:
                    faces.append((x, y, w, h))
        
        return np.array(faces)
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return np.array([])

# Draw rectangle around face with label
def draw_face_rectangle(image, faces, label="Face Detected"):
    image_with_rect = image.copy()
    for (x, y, w, h) in faces:
        # Draw rectangle with thicker line
        cv2.rectangle(image_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Add background for text
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_with_rect, (x, y-30), (x + label_size[0], y), (0, 255, 0), -1)
        
        # Add text
        cv2.putText(image_with_rect, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image_with_rect

# Improved face matching with multiple methods
def improved_face_match(test_image, registered_images, threshold=0.70):
    """
    Improved face matching using multiple comparison methods with YOLOv8 detection
    Returns list of (user_id, name, similarity) tuples
    """
    matches = []
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_faces = detect_faces(test_image)
    
    if len(test_faces) == 0:
        return matches
    
    # Get the first detected face (largest one)
    test_faces = sorted(test_faces, key=lambda x: x[2]*x[3], reverse=True)
    (x, y, w, h) = test_faces[0]
    
    # Add boundary checks
    h_img, w_img = test_gray.shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    
    test_face = test_gray[y:y+h, x:x+w]
    test_face = cv2.resize(test_face, (100, 100))
    
    # Enhance the test face
    test_face = cv2.equalizeHist(test_face)
    
    # Compare with registered faces
    for user_id, img_path in registered_images.items():
        if os.path.exists(img_path):
            try:
                reg_image = cv2.imread(img_path)
                reg_gray = cv2.cvtColor(reg_image, cv2.COLOR_BGR2GRAY)
                reg_faces = detect_faces(reg_image)
                
                if len(reg_faces) > 0:
                    # Get the largest face
                    reg_faces = sorted(reg_faces, key=lambda x: x[2]*x[3], reverse=True)
                    (rx, ry, rw, rh) = reg_faces[0]
                    
                    # Add boundary checks
                    h_reg, w_reg = reg_gray.shape
                    rx = max(0, rx)
                    ry = max(0, ry)
                    rw = min(rw, w_reg - rx)
                    rh = min(rh, h_reg - ry)
                    
                    reg_face = reg_gray[ry:ry+rh, rx:rx+rw]
                    reg_face = cv2.resize(reg_face, (100, 100))
                    
                    # Enhance the registered face
                    reg_face = cv2.equalizeHist(reg_face)
                    
                    # Method 1: Template matching with CCOEFF_NORMED
                    result1 = cv2.matchTemplate(test_face, reg_face, cv2.TM_CCOEFF_NORMED)
                    similarity1 = result1[0][0]
                    
                    # Method 2: Template matching with CCORR_NORMED
                    result2 = cv2.matchTemplate(test_face, reg_face, cv2.TM_CCORR_NORMED)
                    similarity2 = result2[0][0]
                    
                    # Method 3: Histogram comparison
                    hist_test = cv2.calcHist([test_face], [0], None, [256], [0, 256])
                    hist_reg = cv2.calcHist([reg_face], [0], None, [256], [0, 256])
                    similarity3 = cv2.compareHist(hist_test, hist_reg, cv2.HISTCMP_CORREL)
                    
                    # Combined similarity score (weighted average)
                    similarity = (similarity1 * 0.5 + similarity2 * 0.3 + similarity3 * 0.2)
                    
                    if similarity > threshold:
                        # Get user name from database
                        user_row = st.session_state.users_df[st.session_state.users_df['ID'] == user_id].iloc[0]
                        matches.append((user_id, user_row['Name'], similarity))
            except Exception as e:
                st.warning(f"Error processing user {user_id}: {str(e)}")
                continue
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .user-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
    }
    .detected-user-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📸 AI Face Detection Attendance System")
st.markdown("### Automated attendance tracking with YOLOv8 facial detection technology")
st.divider()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/face-id.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["🏠 Dashboard", "➕ Register User", "📸 Mark Attendance", "📊 View Records"])
    
    st.divider()
    st.markdown("### System Info")
    st.success("🤖 **AI Model:** YOLOv8")
    
    # Reload users to show latest count
    if os.path.exists(USERS_FILE):
        current_users = pd.read_csv(USERS_FILE)
        st.info(f"**📊 Registered Users:** {len(current_users)}")
    else:
        st.info(f"**📊 Registered Users:** 0")
    
    df = load_attendance()
    if not df.empty:
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = len(df[df['Date'] == today])
        st.success(f"**✅ Today's Attendance:** {today_count}")
    else:
        st.success(f"**✅ Today's Attendance:** 0")
    
    st.divider()
    st.info("✨ **YOLOv8 Detection**\nAdvanced AI-powered face detection")

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("📊 Dashboard Overview")
    
    # Force reload of data
    if os.path.exists(USERS_FILE):
        st.session_state.users_df = pd.read_csv(USERS_FILE)
    
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Users", len(st.session_state.users_df))
    
    with col2:
        today_count = len(df[df['Date'] == today]) if not df.empty else 0
        st.metric("✅ Present Today", today_count)
    
    with col3:
        total_records = len(df) if not df.empty else 0
        st.metric("📝 Total Records", total_records)
    
    with col4:
        if len(st.session_state.users_df) > 0:
            attendance_rate = (today_count / len(st.session_state.users_df)) * 100
        else:
            attendance_rate = 0
        st.metric("📈 Attendance Rate", f"{attendance_rate:.1f}%")
    
    st.divider()
    
    # Show registered users
    if len(st.session_state.users_df) > 0:
        st.subheader("👥 Recently Registered Users")
        
        # Show last 5 users
        recent_users = st.session_state.users_df.tail(5).sort_values('RegisterDate', ascending=False)
        
        cols = st.columns(5)
        for idx, (_, row) in enumerate(recent_users.iterrows()):
            if idx < 5:
                with cols[idx]:
                    st.markdown(f"**{row['Name']}**")
                    st.caption(f"ID: {row['ID']}")
                    if os.path.exists(row['Photo_Path']):
                        image = cv2.imread(row['Photo_Path'])
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.image("https://img.icons8.com/clouds/200/000000/user.png")
                    st.caption(f"📅 {row['RegisterDate']}")
        
        st.divider()
    
    # Recent Activity
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🕐 Recent Activity")
        if not df.empty:
            recent = df.tail(10).sort_values('Date', ascending=False)
            
            # Styled display
            for idx, row in recent.iterrows():
                with st.container():
                    cols = st.columns([3, 2, 2, 1])
                    cols[0].write(f"**{row['Name']}**")
                    cols[1].write(f"📅 {row['Date']}")
                    cols[2].write(f"⏰ {row['Time']}")
                    cols[3].write(f"✅ {row['Status']}")
                st.divider()
        else:
            st.info("No attendance records yet. Start by registering users!")
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        if not df.empty:
            # Most punctual user
            if 'Time' in df.columns:
                df_copy = df.copy()
                df_copy['Time'] = pd.to_datetime(df_copy['Time'], format='%H:%M:%S', errors='coerce')
                avg_times = df_copy.groupby('Name')['Time'].mean()
                if not avg_times.empty:
                    earliest = avg_times.idxmin()
                    st.success(f"🏆 Most Punctual\n\n**{earliest}**")
            
            # Most attendance
            most_present = df['Name'].value_counts().head(1)
            if not most_present.empty:
                st.info(f"⭐ Top Attendee\n\n**{most_present.index[0]}**\n\n{most_present.values[0]} days")
        else:
            st.info("Statistics will appear here once attendance is recorded")
    
    # Charts
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Weekly Attendance Trend")
        if not df.empty:
            df_chart = df.copy()
            df_chart['Date'] = pd.to_datetime(df_chart['Date'])
            week_data = df_chart[df_chart['Date'] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
            if not week_data.empty:
                daily_counts = week_data.groupby('Date').size().reset_index(name='Count')
                st.bar_chart(daily_counts.set_index('Date'))
            else:
                st.info("No data for the past week")
        else:
            st.info("No data available")
    
    with col2:
        st.subheader("👤 Top 5 Attendees")
        if not df.empty:
            top_users = df['Name'].value_counts().head(5)
            st.bar_chart(top_users)
        else:
            st.info("No data available")

# Register User Page
elif page == "➕ Register User":
    st.header("➕ Register New User")
    st.markdown("*Add new users to the attendance system with YOLOv8 face detection*")
    
    # Show success message if just registered
    if st.session_state.registration_success:
        st.success("✅ User registered successfully! Check the dashboard to see the new user.")
        st.session_state.registration_success = False
    
    # Registration method selection
    upload_option = st.radio("Registration Method", 
                            ["📷 Capture from Webcam", "📁 Upload Image"],
                            horizontal=True)
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 User Information")
        
        # User input fields
        user_id = st.text_input("User ID *", placeholder="e.g., EMP001, STU2024001", key="reg_user_id")
        user_name = st.text_input("Full Name *", placeholder="e.g., John Doe", key="reg_user_name")
    
    with col2:
        st.subheader("📸 Photo Capture")
        
        if upload_option == "📁 Upload Image":
            uploaded_file = st.file_uploader("Choose a clear photo", type=['jpg', 'jpeg', 'png'], key="upload_file")
            
            if uploaded_file:
                # Read file once and store
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.temp_image = image
                
                # Detect face using YOLOv8
                with st.spinner("🤖 Detecting face with YOLOv8..."):
                    faces = detect_faces(image)
                st.session_state.detected_faces = faces
                
                if len(faces) > 0:
                    st.success(f"✅ Face detected! ({len(faces)} face(s) found)")
                    
                    # Show preview with rectangle
                    image_with_rect = draw_face_rectangle(image, faces)
                    st.image(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB), 
                           caption="YOLOv8 Face Detection", use_column_width=True)
                else:
                    st.warning("⚠️ No face detected in the image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                           caption="Uploaded Image", use_column_width=True)
            else:
                st.session_state.temp_image = None
                st.session_state.detected_faces = []
        
        elif upload_option == "📷 Capture from Webcam":
            st.info("💡 Tip: Ensure good lighting and look directly at the camera")
            camera_image = st.camera_input("Take a picture", key="reg_camera")
            
            if camera_image:
                # Read camera image once and store
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.temp_image = image
                
                # Detect face using YOLOv8
                with st.spinner("🤖 Detecting face with YOLOv8..."):
                    faces = detect_faces(image)
                st.session_state.detected_faces = faces
                
                if len(faces) > 0:
                    st.success(f"✅ Face detected! ({len(faces)} face(s) found)")
                    
                    # Show preview with rectangle
                    image_with_rect = draw_face_rectangle(image, faces)
                    st.image(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB), 
                           caption="YOLOv8 Face Detection", use_column_width=True)
                else:
                    st.warning("⚠️ No face detected. Please try again with better lighting.")
            else:
                st.session_state.temp_image = None
                st.session_state.detected_faces = []
    
    # Register button - ALWAYS VISIBLE
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Register User", type="primary", use_container_width=True, key="register_btn"):
            # Validate all fields
            errors = []
            
            if not user_id or not user_id.strip():
                errors.append("❌ User ID is required")
            
            if not user_name or not user_name.strip():
                errors.append("❌ Full Name is required")
            
            if st.session_state.temp_image is None:
                errors.append("❌ Please capture or upload a photo")
            else:
                if len(st.session_state.detected_faces) == 0:
                    errors.append("❌ No face detected in the image. Please try again with a clearer photo")
            
            # Check if user already exists
            if user_id and user_id.strip() and user_id in st.session_state.users_df['ID'].values:
                errors.append("❌ User ID already exists! Please use a different ID")
            
            # Show errors or register
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Save image
                image_path = f"images/{user_id}.jpg"
                cv2.imwrite(image_path, st.session_state.temp_image)
                
                # Add user to database
                new_user = pd.DataFrame([{
                    'ID': user_id,
                    'Name': user_name,
                    'Photo_Path': image_path,
                    'RegisterDate': datetime.now().strftime("%Y-%m-%d")
                }])
                
                st.session_state.users_df = pd.concat([st.session_state.users_df, new_user], ignore_index=True)
                save_users()
                
                # Clear temp image
                st.session_state.temp_image = None
                st.session_state.detected_faces = []
                
                # Set success flag
                st.session_state.registration_success = True
                
                st.success(f"✅ User {user_name} (ID: {user_id}) registered successfully!")
                st.balloons()
                
                # Show success message with user details
                st.info(f"""
                **Registration Details:**
                - **Name:** {user_name}
                - **ID:** {user_id}
                - **Date:** {datetime.now().strftime("%Y-%m-%d")}
                - **Photo saved:** {image_path}
                - **Detection:** YOLOv8
                """)
                
                st.success("✅ You can now mark attendance for this user using YOLOv8 face detection!")
                
                # Show current total
                st.metric("Total Registered Users", len(st.session_state.users_df))
                
                # Suggest next step
                st.info("👉 Navigate to '📸 Mark Attendance' to mark attendance using YOLOv8 face detection.")

# Mark Attendance Page
elif page == "📸 Mark Attendance":
    st.header("📸 Mark Attendance")
    st.markdown("*Automatic attendance marking using YOLOv8 face detection*")
    
    # Reload users
    if os.path.exists(USERS_FILE):
        st.session_state.users_df = pd.read_csv(USERS_FILE)
    
    if len(st.session_state.users_df) == 0:
        st.warning("⚠️ No users registered yet. Please register users first.")
        st.info("👈 Use the sidebar to navigate to 'Register User' page.")
    else:
        st.info("📸 **YOLOv8 Face Detection**: Scan your face to automatically mark attendance")
        st.success("✨ **How it works**: Just capture your photo - YOLOv8 will identify you and mark attendance!")
        
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Scan Your Face")
            
            st.info("💡 Look directly at the camera for best results")
            camera_image = st.camera_input("Take a picture", key="attendance_camera")
            
            attendance_image = None
            
            if camera_image:
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                attendance_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process the image
            if attendance_image is not None:
                with st.spinner("🤖 Detecting face with YOLOv8..."):
                    faces = detect_faces(attendance_image)
                
                if len(faces) > 0:
                    st.success(f"✅ {len(faces)} face(s) detected by YOLOv8!")
                    
                    # Create a mapping of registered users
                    registered_images = {}
                    for _, row in st.session_state.users_df.iterrows():
                        registered_images[row['ID']] = row['Photo_Path']
                    
                    # Try to match faces
                    with st.spinner("🔍 Identifying person from database..."):
                        matches = improved_face_match(attendance_image, registered_images, threshold=0.65)
                    
                    if matches:
                        # Get the best match (first one, highest similarity)
                        user_id, user_name, similarity = matches[0]
                        
                        # Show image with detected face and name
                        image_with_label = draw_face_rectangle(attendance_image, faces, 
                                                               f"{user_name} ({user_id})")
                        st.image(cv2.cvtColor(image_with_label, cv2.COLOR_BGR2RGB), 
                               caption="Face Identified by YOLOv8!", use_column_width=True)
                        
                        st.divider()
                        
                        # Display detected user info prominently
                        st.markdown(f"""
                        <div class="detected-user-box">
                            <h2>🎯 Detected User (YOLOv8)</h2>
                            <h3>👤 Name: {user_name}</h3>
                            <h3>🆔 User ID: {user_id}</h3>
                            <h4>📊 Match Confidence: {similarity*100:.1f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # AUTOMATICALLY MARK ATTENDANCE
                        success, message = mark_attendance(user_id, user_name)
                        
                        if success:
                            st.success(f"🎉 **Attendance Marked Successfully!**")
                            st.balloons()
                            
                            # Display detailed confirmation
                            with st.container():
                                col_a, col_b = st.columns([1, 1])
                                
                                with col_a:
                                    st.markdown("### ✅ Attendance Confirmed")
                                    st.write(f"**Name:** {user_name}")
                                    st.write(f"**ID:** {user_id}")
                                    st.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
                                    st.write(f"**Match Score:** {similarity*100:.1f}%")
                                    st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
                                    st.write(f"**Detection:** YOLOv8")
                                
                                with col_b:
                                    # Show registered photo
                                    user_row = st.session_state.users_df[st.session_state.users_df['ID'] == user_id].iloc[0]
                                    if os.path.exists(user_row['Photo_Path']):
                                        reg_image = cv2.imread(user_row['Photo_Path'])
                                        st.image(cv2.cvtColor(reg_image, cv2.COLOR_BGR2RGB), 
                                               caption="Registered Photo", use_column_width=True)
                            
                            st.divider()
                            st.info("💡 **Attendance recorded in the system!** Check 'View Records' to see all attendance.")
                            
                            # Show Excel file download option
                            if os.path.exists(ATTENDANCE_EXCEL_FILE):
                                with open(ATTENDANCE_EXCEL_FILE, 'rb') as f:
                                    excel_data = f.read()
                                st.download_button(
                                    label="📥 Download Today's Attendance (Excel)",
                                    data=excel_data,
                                    file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    type="primary"
                                )
                            
                        else:
                            # Already marked today
                            st.warning(f"⚠️ **{message}**")
                            
                            with st.container():
                                col_a, col_b = st.columns([1, 1])
                                
                                with col_a:
                                    st.markdown("### 👤 User Identified")
                                    st.write(f"**Name:** {user_name}")
                                    st.write(f"**ID:** {user_id}")
                                    st.write(f"**Match Score:** {similarity*100:.1f}%")
                                    st.info("Your attendance has already been recorded for today.")
                                
                                with col_b:
                                    user_row = st.session_state.users_df[st.session_state.users_df['ID'] == user_id].iloc[0]
                                    if os.path.exists(user_row['Photo_Path']):
                                        reg_image = cv2.imread(user_row['Photo_Path'])
                                        st.image(cv2.cvtColor(reg_image, cv2.COLOR_BGR2RGB), 
                                               caption="Registered Photo", use_column_width=True)
                        
                        # Show if there are additional matches
                        if len(matches) > 1:
                            st.divider()
                            st.info(f"ℹ️ Found {len(matches)-1} additional similar face(s) in database")
                            with st.expander("View other possible matches"):
                                for i, (uid, uname, sim) in enumerate(matches[1:], 1):
                                    st.write(f"{i}. **{uname}** (ID: {uid}) - Similarity: {sim*100:.1f}%")
                            
                    else:
                        st.error("❌ **Face Not Recognized**")
                        
                        # Show the image
                        image_with_rect = draw_face_rectangle(attendance_image, faces, "Unknown")
                        st.image(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB), 
                               caption="Face Detected by YOLOv8 but Not Recognized", use_column_width=True)
                        
                        st.warning("⚠️ No matching faces found in the database")
                        st.info("""
                        **Possible reasons:**
                        - You may not be registered in the system
                        - Photo quality may be insufficient
                        - Lighting conditions are poor
                        - Face angle is too different from registration photo
                        
                        **Solutions:**
                        - Ensure good lighting
                        - Face the camera directly
                        - Remove glasses/masks if possible
                        - Ask admin to register you first
                        - Try capturing the photo again
                        """)
                else:
                    st.error("❌ **No Face Detected by YOLOv8**")
                    st.image(cv2.cvtColor(attendance_image, cv2.COLOR_BGR2RGB), 
                           caption="No Face Detected", use_column_width=True)
                    st.warning("⚠️ YOLOv8 could not detect a face in the image")
                    st.info("""
                    **Tips for better detection:**
                    - Ensure good lighting
                    - Look directly at the camera
                    - Remove obstructions (hat, mask, etc.)
                    - Move closer to the camera
                    - Try capturing the photo again
                    """)
        
        with col2:
            st.subheader("📊 Quick Info")
            
            # Show registered users count
            st.metric("👥 Registered Users", len(st.session_state.users_df))
            
            # Show today's attendance
            df = load_attendance()
            today = datetime.now().strftime("%Y-%m-%d")
            
            if not df.empty:
                today_records = df[df['Date'] == today]
                st.metric("✅ Present Today", len(today_records))
                
                if not today_records.empty:
                    st.write("**Today's Attendance:**")
                    for _, row in today_records.iterrows():
                        st.write(f"• {row['Name']} ({row['Time']})")
            else:
                st.metric("✅ Present Today", 0)
            
            st.divider()
            st.info("🤖 **YOLOv8 Detection:**\n\n• State-of-the-art AI\n• Fast & accurate\n• Robust detection\n• Real-time processing")
        
        # Today's attendance summary
        st.divider()
        st.subheader("📋 Today's Attendance Summary")
        df = load_attendance()
        today = datetime.now().strftime("%Y-%m-%d")
        
        if not df.empty:
            today_records = df[df['Date'] == today]
            if not today_records.empty:
                st.dataframe(today_records[['ID', 'Name', 'Time', 'Status']], 
                           use_container_width=True, hide_index=True)
            else:
                st.info("No attendance marked today yet.")
        else:
            st.info("No attendance records available.")

# View Records Page
elif page == "📊 View Records":
    st.header("📊 Attendance Records & Analytics")
    
    # Reload users
    if os.path.exists(USERS_FILE):
        st.session_state.users_df = pd.read_csv(USERS_FILE)
    
    tab1, tab2, tab3 = st.tabs(["📋 All Records", "👥 Registered Users", "📈 Analytics"])
    
    with tab1:
        st.subheader("📋 Attendance History")
        df = load_attendance()
        
        if not df.empty:
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_date = st.date_input("📅 Filter by Date", value=None)
            
            with col2:
                filter_name = st.selectbox("👤 Filter by Name", ["All"] + sorted(df['Name'].unique().tolist()))
            
            with col3:
                filter_status = st.selectbox("📊 Status", ["All", "Present", "Absent"])
            
            with col4:
                st.write("")
                st.write("")
                # Excel download button
                if os.path.exists(ATTENDANCE_EXCEL_FILE):
                    with open(ATTENDANCE_EXCEL_FILE, 'rb') as f:
                        excel_data = f.read()
                    st.download_button(
                        "📥 Download Excel",
                        data=excel_data,
                        file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.download_button(
                        "📥 Download CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Apply filters
            filtered_df = df.copy()
            
            if filter_date:
                filtered_df = filtered_df[filtered_df['Date'] == filter_date.strftime("%Y-%m-%d")]
            
            if filter_name != "All":
                filtered_df = filtered_df[filtered_df['Name'] == filter_name]
            
            if filter_status != "All":
                filtered_df = filtered_df[filtered_df['Status'] == filter_status]
            
            # Display table
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Statistics
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Records", len(filtered_df))
            
            with col2:
                unique_users = filtered_df['Name'].nunique()
                st.metric("👥 Unique Users", unique_users)
            
            with col3:
                if not filtered_df.empty:
                    unique_dates = filtered_df['Date'].nunique()
                    st.metric("📅 Days Covered", unique_dates)
        else:
            st.info("📝 No attendance records available yet. Start marking attendance!")
    
    with tab2:
        st.subheader("👥 Registered Users Database")
        
        if len(st.session_state.users_df) > 0:
            # Display user table
            st.dataframe(
                st.session_state.users_df[['ID', 'Name', 'RegisterDate']], 
                use_container_width=True, 
                hide_index=True
            )
            
            # Show user photos in grid
            st.divider()
            st.subheader("📸 User Photos Gallery")
            
            cols = st.columns(4)
            for idx, row in st.session_state.users_df.iterrows():
                col_idx = idx % 4
                with cols[col_idx]:
                    st.markdown(f"**{row['Name']}**")
                    st.caption(f"ID: {row['ID']}")
                    if os.path.exists(row['Photo_Path']):
                        image = cv2.imread(row['Photo_Path'])
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.info("No photo")
                    st.markdown("---")
            
            # Manage users
            st.divider()
            st.subheader("⚙️ User Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                delete_user = st.selectbox("Select user to delete", 
                                          ["None"] + st.session_state.users_df['ID'].tolist())
            
            with col2:
                st.write("")
                st.write("")
                if delete_user != "None":
                    if st.button("🗑️ Delete User", type="secondary", use_container_width=True):
                        # Remove from dataframe
                        user_row = st.session_state.users_df[st.session_state.users_df['ID'] == delete_user].iloc[0]
                        
                        # Delete image
                        if os.path.exists(user_row['Photo_Path']):
                            os.remove(user_row['Photo_Path'])
                        
                        st.session_state.users_df = st.session_state.users_df[st.session_state.users_df['ID'] != delete_user]
                        save_users()
                        
                        st.success("✅ User deleted successfully!")
                        st.rerun()
        else:
            st.info("👤 No users registered yet. Go to 'Register User' to add users.")
    
    with tab3:
        st.subheader("📈 Attendance Analytics")
        
        df = load_attendance()
        
        if not df.empty and len(st.session_state.users_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Attendance by user
                st.markdown("**📊 Attendance Count by User**")
                user_counts = df['Name'].value_counts()
                st.bar_chart(user_counts)
            
            with col2:
                # Monthly trend
                st.markdown("**📅 Monthly Attendance Trend**")
                df_copy = df.copy()
                df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                df_copy['Month'] = df_copy['Date'].dt.to_period('M').astype(str)
                monthly_counts = df_copy.groupby('Month').size()
                st.line_chart(monthly_counts)
            
            st.divider()
            
            # Detailed stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_daily = len(df) / df['Date'].nunique()
                st.metric("📊 Avg Daily Attendance", f"{avg_daily:.1f}")
            
            with col2:
                total_days = df['Date'].nunique()
                st.metric("📅 Total Active Days", total_days)
            
            with col3:
                completion_rate = (len(df) / (len(st.session_state.users_df) * total_days)) * 100
                st.metric("✅ Completion Rate", f"{completion_rate:.1f}%")
            
        else:
            st.info("📊 Analytics will appear here once you have attendance data.")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Face Detection Attendance System v4.0 - YOLOv8 Edition</strong></p>
        <p>Built with Streamlit, OpenCV & YOLOv8 | Advanced AI Face Recognition</p>
        <p><i>🤖 Powered by YOLOv8 for superior face detection accuracy</i></p>
    </div>
""", unsafe_allow_html=True)