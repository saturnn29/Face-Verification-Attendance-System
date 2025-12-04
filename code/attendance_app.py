import customtkinter as ctk
from customtkinter import CTkInputDialog, CTkImage
import cv2
from PIL import Image, ImageTk
import threading
import datetime
from deepface import DeepFace
from face_recognition_backend import FaceRecognitionBackend
import numpy as np

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Checking System")
        self.root.geometry("900x700")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.webcam_active = False
        self.cap = None
        self.recognition_backend = FaceRecognitionBackend()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.setup_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Set up the user interface"""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        title_label = ctk.CTkLabel(
            self.root,
            text="Attendance Checking System",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=20, padx=20, sticky="ew")
        
        main_frame = ctk.CTkFrame(self.root)
        main_frame.grid(row=1, column=0, pady=10, padx=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(
            main_frame,
            text="Camera Feed\n(Click 'Start Camera' to begin)",
            width=640,
            height=480,
            fg_color="gray20",
            font=ctk.CTkFont(size=16)
        )
        self.video_label.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")
        
        control_frame = ctk.CTkFrame(self.root)
        control_frame.grid(row=2, column=0, pady=10, padx=20, sticky="ew")
        control_frame.grid_columnconfigure(0, weight=1)
        
        self.camera_button = ctk.CTkButton(
            control_frame,
            text="Start Camera",
            command=self.toggle_camera,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.camera_button.grid(row=0, column=0, pady=10, padx=20)
        
        self.verify_button = ctk.CTkButton(
            control_frame,
            text="Verify Attendance",
            command=self.verify_attendance,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40,
            state="disabled"
        )
        self.verify_button.grid(row=0, column=1, pady=10, padx=20)
        
        self.enroll_button = ctk.CTkButton(
            control_frame,
            text="Enroll New Employee",
            command=self.enroll_new_employee,  # <-- This is a new function
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40,
            state="disabled",  # Will be enabled when camera starts
            fg_color="blue",
            hover_color="darkblue"
        )
        self.enroll_button.grid(row=0, column=2, pady=10, padx=20)
        
        message_frame = ctk.CTkFrame(self.root)
        message_frame.grid(row=3, column=0, pady=10, padx=20, sticky="ew")
        message_frame.grid_columnconfigure(0, weight=1)
        
        message_label = ctk.CTkLabel(
            message_frame,
            text="Status Messages:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        message_label.grid(row=0, column=0, pady=(10, 5), padx=20, sticky="w")
        
        self.message_display = ctk.CTkTextbox(
            message_frame,
            height=100,
            font=ctk.CTkFont(size=13),
            state="disabled"
        )
        self.message_display.grid(row=1, column=0, pady=(0, 10), padx=20, sticky="ew")
        
        self.add_message("System ready. Click 'Start Camera' to begin.")
    
    def toggle_camera(self):
        """Toggle the webcam on/off"""
        if not self.webcam_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the webcam"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.add_message("ERROR: Unable to access webcam. Please check your camera.", "error")
            return
        
        self.webcam_active = True
        self.camera_button.configure(text="Stop Camera", fg_color="red", hover_color="darkred")
        self.verify_button.configure(state="normal")
        self.enroll_button.configure(state="normal")
        self.add_message("Camera started successfully.")
        
        self.update_frame()
    
    def stop_camera(self):
        """Stop the webcam"""
        self.webcam_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_button.configure(text="Start Camera", fg_color="green", hover_color="darkgreen")
        self.verify_button.configure(state="disabled")
        self.enroll_button.configure(state="disabled")
        
        self.video_label.configure(
            image=None,
            text="Camera Feed\n(Click 'Start Camera' to begin)"
        )
        self.video_label._image = None
        
        self.add_message("Camera stopped.")
    
    def update_frame(self):
        """Update the video frame"""
        if self.webcam_active and self.cap:
            ret, frame = self.cap.read()
            
            if ret:
                # 1. Update liveness buffer (this is unchanged)
                self.recognition_backend.update_liveness_buffer(frame)

                # 2. Convert to gray for detectors
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 3. Find all faces in the frame
                faces = self.face_cascade.detectMultiScale(
                    gray_frame, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(60, 60)
                )
                
                # --- NEW: Process each face for emotion ---
                for (x, y, w, h) in faces:
                    # Draw bounding box (Green)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    emotion_label = "..." # Default text while processing
                    
                    # --- Emotion detection starts here (using DeepFace) ---
                    try:
                        # 1. Get the color face region
                        face_roi_color = frame[y:y+h, x:x+w]

                        # 2. Analyze it directly
                        # We use enforce_detection=False because we already have the face
                        analysis = DeepFace.analyze(
                            face_roi_color, 
                            actions=['emotion'], 
                            enforce_detection=False
                        )

                        # 3. Get the label text
                        # Capitalize it to look nice (e.g., "happy" -> "Happy")
                        emotion_label = analysis[0]['dominant_emotion'].capitalize()

                    except Exception as e:
                        # This will fail on the first frame or if face is unclear
                        # print(f"Error during live emotion detection: {e}")
                        emotion_label = "..." # Keep it simple
                # --- Emotion detection ends here ---

                    # --- Draw the text on the *color* frame ---
                    text_pos = (x, y - 10) # Position text just above the box
                    cv2.putText(
                        frame, 
                        emotion_label, 
                        text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7,           # Font scale
                        (0, 255, 0),   # Font color (Green)
                        2              # Font thickness
                    )
                # --- END NEW ---

                # Convert the frame (with box and text) for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                
                img = Image.fromarray(frame_resized)
                # 1. Create the image
                ctk_img = CTkImage(img, size=(640, 480))
                
                # 2. Configure the label to use it
                self.video_label.configure(image=ctk_img, text="")
                
                # 3. CRITICAL: Store a reference on the widget itself
                # This prevents the image from being garbage-collected.
                self.video_label.image = ctk_img
            
            # This is recursive, it calls itself
            self.root.after(10, self.update_frame)
    
    def verify_attendance(self):
        """Verify attendance using the current frame"""
        if not self.webcam_active or not self.cap:
            self.add_message("ERROR: Camera is not active.", "error")
            return
        
        ret, frame = self.cap.read()
        
        if not ret:
            self.add_message("ERROR: Unable to capture frame from camera.", "error")
            return
        
        self.add_message("Verifying attendance...")
        
        def verification_thread():
            result = self.recognition_backend.verify_employee(frame)
            
            self.root.after(0, lambda: self.display_verification_result(result))
        
        thread = threading.Thread(target=verification_thread, daemon=True)
        thread.start()
    
    def display_verification_result(self, result):
        """Display the verification result"""
        if result['success']:
            employee = result['employee']
            message = f"✓ {result['message']}\n"
            if employee:
                message += f"  ID: {employee.get('id', 'N/A')}\n"
                message += f"  Department: {employee.get('department', 'N/A')}"
            self.add_message(message, "success")
        else:
            self.add_message(f"✗ {result['message']}", "error")
    
    def add_message(self, message, message_type="info"):
        """Add a message to the message display"""
        self.message_display.configure(state="normal")
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if message_type == "error":
            formatted_message = f"[{timestamp}] ❌ {message}\n"
        elif message_type == "success":
            formatted_message = f"[{timestamp}] ✅ {message}\n"
        else:
            formatted_message = f"[{timestamp}] ℹ️  {message}\n"
        
        self.message_display.insert("end", formatted_message)
        self.message_display.see("end")
        self.message_display.configure(state="disabled")
        
    def enroll_new_employee(self):
        """Enroll a new employee using pop-up dialogs."""
        if not self.webcam_active or not self.cap:
            self.add_message("ERROR: Camera must be active to enroll.", "error")
            return

        # 1. Get Employee ID
        dialog = CTkInputDialog(text="Enter new Employee ID (e.g., EMP_003):", title="Enrollment - Step 1 of 3")
        emp_id = dialog.get_input()

        if not emp_id:
            self.add_message("Enrollment cancelled.", "info")
            return

        # 2. Get Employee Name
        dialog = CTkInputDialog(text="Enter Employee's Full Name:", title="Enrollment - Step 2 of 3")
        emp_name = dialog.get_input()

        if not emp_name:
            self.add_message("Enrollment cancelled.", "info")
            return

        # 3. Get Employee Department
        dialog = CTkInputDialog(text="Enter Employee's Department:", title="Enrollment - Step 3 of 3")
        emp_dept = dialog.get_input()

        if not emp_dept:
            self.add_message("Enrollment cancelled.", "info")
            return

        # 4. Capture and Process Face
        self.add_message(f"Capturing face for {emp_name}... Please look at the camera.", "info")

        ret, frame = self.cap.read()
        if not ret:
            self.add_message("ERROR: Unable to capture frame from camera.", "error")
            return

        # Send to backend for processing in a thread
        def enroll_thread():
            result = self.recognition_backend.enroll_employee(frame, emp_id, emp_name, emp_dept)

            # Display result back in the main thread
            self.root.after(0, lambda: self.display_enrollment_result(result))

        threading.Thread(target=enroll_thread, daemon=True).start()

    def display_enrollment_result(self, result):
        """Displays the result from the backend enrollment function."""
        if result['success']:
            self.add_message(f"✓ {result['message']}", "success")
        else:
            self.add_message(f"✗ {result['message']}", "error")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()
