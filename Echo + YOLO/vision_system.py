import cv2
import threading
import time
import logging
import os
import numpy as np
import json
from collections import deque, Counter
from ultralytics import YOLO
from shared.state import SharedState
from object_tracker import ObjectTracker
from vision_learning import ObjectLearner

# Try to import picamera2 for CSI camera support (Raspberry Pi)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class VisionSystem:
    def __init__(self, shared_state: SharedState, model_path="yolov8n.pt"):
        self.shared_state = shared_state
        self.model_path = model_path
        self.running = False
        self.thread = None
        self.tracker = ObjectTracker()
        self.learner = ObjectLearner()
        self.cap = None
        self.learner = ObjectLearner()
        self.cap = None
        self.latest_frame = None # Store latest frame for learning
        self.learning_lock = threading.Lock() # Lock for learning operations
        
        # Face Recognition (Lite / LBPH)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        try:
             self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
             self.face_recognizer = cv2.face_LBPHFaceRecognizer.create()
        self.faces_db_file = "faces_db.json"
        self.face_model_file = "face_model.yml"
        self.known_names = {} # ID (int) -> Name (str)
        self.training_mode = False
        self.training_name = None
        self.training_samples = []
        self.next_id = 1
        
        # Smoothing / History Buffer
        # Store last 10 recognitions to prevent flickering
        self.recognition_history = deque(maxlen=10) 
        
        self._load_face_data()

    def _load_face_data(self):
        """Loads face DB and model."""
        if os.path.exists(self.faces_db_file):
            try:
                with open(self.faces_db_file, 'r') as f:
                    data = json.load(f)
                    self.known_names = {int(k): v for k, v in data.items()}
                    if self.known_names:
                        self.next_id = max(self.known_names.keys()) + 1
            except Exception as e:
                logging.error(f"❌ Failed to load faces DB: {e}")
        
        if os.path.exists(self.face_model_file):
            try:
                self.face_recognizer.read(self.face_model_file)
                logging.info(f"👤 Loaded face model with {len(self.known_names)} people.")
            except Exception as e:
                logging.error(f"❌ Failed to load face model: {e}") 

    def start_registration(self, name: str):
        """Starts the face training process."""
        logging.info(f"👤 Starting face registration for: {name}")
        with self.learning_lock:
            self.training_name = name
            self.training_samples = []
            self.training_mode = True
        return f"Look at the camera! I'm learning {name}'s face..."

    def forgetting_person(self, name: str):
        """Removes a person from the DB (requires retraining from scratch practically, but for now just removing name)."""
        # Note: LBPH cannot easily 'forget' one person without retraining on all raw images. 
        # A simple hack is just to remove the name from the map so ID maps to Unknown, 
        # but the model will still predict that ID. 
        # Proper way: Save all raw images, delete user folder, retrain. 
        # For 'Lite' version: We will just remove the name mapping.
        to_delete = [id for id, n in self.known_names.items() if n.lower() == name.lower()]
        for id in to_delete:
            del self.known_names[id]
        
        # Save DB
        try:
            with open(self.faces_db_file, 'w') as f:
                json.dump(self.known_names, f)
            return f"Forgot {name}."
        except Exception as e:
            return f"Error saving DB: {e}"

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logging.info("👁️ Vision System started.")

    def check_camera(self) -> bool:
        """
        Verifies if the camera is accessible.
        Returns: True if camera works, False otherwise.
        """
        logging.info("👁️ Checking camera connection...")
        
        # Try picamera2 first (for CSI cameras like Arducam)
        if PICAMERA_AVAILABLE:
            try:
                picam = Picamera2()
                config = picam.create_video_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                picam.configure(config)
                time.sleep(0.5)
                picam.start()
                # Try to capture one frame
                time.sleep(0.5)
                frame = picam.capture_array()
                picam.stop()
                picam.close()
                if frame is not None:
                    logging.info("✅ Camera found: Picamera2 (CSI)")
                    return True
            except Exception as e:
                logging.debug(f"Picamera2 check failed: {e}")
                try:
                    if 'picam' in locals():
                        picam.stop()
                        picam.close()
                except:
                    pass
        
        # Fallback to USB cameras
        for i in range(5):
             try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        logging.info(f"✅ Camera found at index {i}")
                        return True
             except:
                 pass
        
        logging.error("❌ No working camera found (checked CSI and USB 0-4).")
        return False

    def learn_current_object(self, name: str):
        """Triggers learning of the object currently in the center of the frame."""
        with self.learning_lock:
            if self.latest_frame is not None:
                return self.learner.learn_object(self.latest_frame, name)
        return False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        logging.info("👁️ Vision System stopped.")

    def _run_loop(self):
        logging.info(f"👁️ Loading YOLO model: {self.model_path}...")
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            logging.error(f"❌ Failed to load YOLO model: {e}")
            self.running = False
            return

        logging.info("👁️ Opening camera...")
        self.cap = None
        self.use_picamera = False
        
        # Try libcamera through OpenCV first (for CSI cameras on Raspberry Pi)
        # This is more reliable than using Picamera2 directly
        try:
            # Try libcamera backend with OpenCV
            cap_test = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
            if cap_test.isOpened():
                # Check if it's actually working
                ret, test_frame = cap_test.read()
                if ret and test_frame is not None:
                    self.cap = cap_test
                    self.use_picamera = False  # We're using OpenCV, not picamera2
                    logging.info("✅ Vision System using CSI camera via OpenCV (/dev/video0)")
                else:
                    cap_test.release()
            else:
                cap_test.release()
        except Exception as e:
            logging.debug(f"⚠️ /dev/video0 failed: {e}")
        
        # Fallback to USB cameras if CSI camera not working
        if self.cap is None:
            for i in range(5):
                try:
                    temp_cap = cv2.VideoCapture(i)
                    if temp_cap.isOpened():
                        self.cap = temp_cap
                        logging.info(f"✅ Vision System using USB camera index {i}")
                        break
                except:
                    continue
                    
            if not self.cap or not self.cap.isOpened():
                logging.error("❌ Could not open any camera (checked /dev/video0 and USB 0-4).")
                self.running = False
                return
                
        # OPTIMIZATION: Set specific lightweight parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Frame skipping for performance (process every Nth frame)
        process_every_n_frames = 5
        frame_count = 0

        while self.running:
            # Capture frame using OpenCV
            ret, frame = self.cap.read()
                
            if not ret or frame is None:
                logging.debug("⚠️ Failed to grab frame.")
                time.sleep(0.1)
                continue
            
            # Small sleep to prevent CPU spinning if camera is fast
            time.sleep(0.01)

            frame_count += 1
            
            # Store frame for potential learning (thread safe copy)
            with self.learning_lock:
                self.latest_frame = frame.copy()
            
            if frame_count % process_every_n_frames != 0:
                continue

            # Run detection
            try:
                results = model(frame, verbose=False, stream=True) 
                
                detected_objects = []
                # stream=True returns a generator, iterate to get results
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        if conf > 0.6: # Confidence threshold increased to reduce false positives
                            detected_objects.append({
                                "label": label,
                                "confidence": conf,
                                "bbox": xyxy
                            })
                            
                # Check for custom objects (center crop)
                # We check this in addition to YOLO
                custom_label = self.learner.identify_object(frame)
                if custom_label:
                    detected_objects.append({
                        "label": custom_label,
                        "confidence": 1.0, # Custom match is treated as high confidence
                        "bbox": [] # No bbox for now, assumed center
                    })
                
                # Update tracker (simplistic for now)
                tracked_objects = self.tracker.update(detected_objects)
                
                # Update shared state
                self.shared_state.update_vision(tracked_objects)

                # --- FACE RECOGNITION BLOCK ---
                # Run periodically or every frame? Every frame is fine for LBPH (fast).
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
                
                if self.training_mode:
                    for (x, y, w, h) in faces:
                        # Save sample
                        self.training_samples.append(gray[y:y+h, x:x+w])
                        
                        # Visual feedback (optional, but we can't see it)
                        if len(self.training_samples) >= 30: # 30 samples
                            # Train
                            # Check if name already exists (case-insensitive)
                            existing_id = None
                            for k, v in self.known_names.items():
                                if v.lower() == self.training_name.lower():
                                    existing_id = k
                                    break
                            
                            if existing_id:
                                new_id = existing_id
                                logging.info(f"👤 Updating existing ID {new_id} for {self.training_name}")
                            else:
                                new_id = self.next_id
                                self.next_id += 1
                                
                            self.known_names[new_id] = self.training_name
                            
                            # If model exists, we should ideally update it. 
                            # LBPH update() allows adding new data!
                            self.face_recognizer.update(self.training_samples, np.array([new_id] * len(self.training_samples)))
                            
                            # Save
                            try:
                                self.face_recognizer.write(self.face_model_file)
                                with open(self.faces_db_file, 'w') as f:
                                    json.dump(self.known_names, f)
                            except Exception as e:
                                logging.error(f"❌ Failed to save face data: {e}")
                                
                            logging.info(f"👤 Learned {self.training_name} (ID: {new_id})")
                            self.training_mode = False
                            self.training_samples = []
                            self.shared_state.update_person(self.training_name) # Confirm immediately
                            break # creating only 1 person at a time

                else: 
                    # --- RECOGNITION MODE ---
                    current_match = None
                    
                    if faces is not None and len(faces) > 0 and self.known_names:
                        for (x, y, w, h) in faces:
                            try:
                                id_, confidence = self.face_recognizer.predict(gray[y:y+h, x:x+w])
                                # confidence is 'distance' (0 is perfect match). 
                                
                                predicted_name = self.known_names.get(id_, "Unknown")
                                
                                # Relaxed threshold to 120 (more forgiving)
                                # Standard is usually ~50-80, but LBPH varies wildly.
                                if confidence < 120:
                                    current_match = predicted_name
                                    # Log match
                                    logging.debug(f"✅ Recognized: {predicted_name} (Dist={confidence:.1f})")
                                else:
                                    logging.debug(f"❌ Missed: {predicted_name} (Dist={confidence:.1f} > 120)")
                                    
                            except Exception as e:
                                pass
                        
                        # Just take the last face found for now
                    
                    # Add result to history
                    self.recognition_history.append(current_match)
                        
                    # --- VOTING / SMOOTHING LOGIC (Always Runs) ---
                    # We need a stable consensus to update the shared state
                    # Only update if history is full enough to be meaningful
                    if len(self.recognition_history) >= 3: # Reduced from 5
                        # Count non-None values
                        valid_votes = [n for n in self.recognition_history if n is not None]
                        if valid_votes:
                            most_common, count = Counter(valid_votes).most_common(1)[0]
                            # Lowered threshold to 30% (3/10 frames) for faster login
                            if count >= len(self.recognition_history) * 0.3:
                                self.shared_state.update_person(most_common)
                            else:
                                self.shared_state.update_person(None)
                        else:
                            self.shared_state.update_person(None)
                    else:
                        # Not enough history, just pass through or wait?
                        if current_match:
                             self.shared_state.update_person(current_match)
                # ------------------------------

            except Exception as e:
                logging.error(f"❌ Error in vision loop: {e}")
                time.sleep(1)

        self.cap.release()
