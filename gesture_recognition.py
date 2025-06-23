import cv2
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SimpleGestureRecognizer:
    def __init__(self):
        # Initialize the ML model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_trained = False

        # Gesture labels
        self.gesture_labels = ['Open_Hand', 'Closed_Fist', 'Peace_Sign', 'Thumbs_Up']

        # Data storage
        self.training_data = []
        self.training_labels = []

        # Background subtractor for hand detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    def preprocess_frame(self, frame):
        """
        Preprocess frame for hand detection using background subtraction
        This is a simpler alternative to MediaPipe
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return blurred

    def extract_hand_contour(self, frame):
        """
        Extract hand contour using background subtraction and contour detection
        Returns: Largest contour (assuming it's the hand) and its features
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (assuming it's the hand)
            largest_contour = max(contours, key=cv2.contourArea)

            # Only process if contour is large enough
            if cv2.contourArea(largest_contour) > 5000:
                return largest_contour, fg_mask

        return None, fg_mask

    def extract_features_from_contour(self, contour):
        features = []

        # 1. Contour area
        area = cv2.contourArea(contour)
        features.append(area)

        # 2. Contour perimeter
        perimeter = cv2.arcLength(contour, True)
        features.append(perimeter)

        # 3. Aspect ratio of bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        features.append(aspect_ratio)

        # 4. Extent (ratio of contour area to bounding rectangle area)
        rect_area = w * h
        extent = float(area) / rect_area
        features.append(extent)

        # 5. Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
        else:
            solidity = 0
        features.append(solidity)

        # 6. Number of convexity defects (fingers)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                # Count significant defects (potential fingers)
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 10000:  # Threshold for significant defects
                        finger_count += 1
                features.append(finger_count)
            else:
                features.append(0)
        else:
            features.append(0)

        # 7. Moments-based features
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            # Centroid
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Normalized central moments (shape descriptors)
            hu_moments = cv2.HuMoments(moments)
            for hu in hu_moments:
                features.append(float(hu[0]))
        else:
            # Add zeros if moments can't be calculated
            features.extend([0, 0] + [0] * 7)

        return features

    def collect_training_data(self):
        print("Starting data collection...")
        print("Instructions:")
        print("1. Keep your hand steady when collecting data")
        print("2. Make sure good lighting and contrasting background")
        print("3. Press 's' to save current gesture, 'n' for next gesture")
        print("4. Press 'q' to quit")

        cap = cv2.VideoCapture(0)
        current_gesture = 0
        samples_per_gesture = 30
        current_samples = 0

        # Let background subtractor learn the background
        print("Learning background... Keep your hand away from camera for 3 seconds")
        for i in range(90):  # 3 seconds at ~30fps
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                processed = self.preprocess_frame(frame)
                self.bg_subtractor.apply(processed)

                cv2.putText(frame, f"Learning background: {3 - i // 30}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1)

        print("Background learned! Now show your gestures.")

        while current_gesture < len(self.gesture_labels):
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Preprocess frame
            processed = self.preprocess_frame(frame)

            # Extract hand contour
            contour, mask = self.extract_hand_contour(processed)

            # Draw contour if detected
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                # Draw bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display instructions
            cv2.putText(frame, f"Gesture: {self.gesture_labels[current_gesture]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {current_samples}/{samples_per_gesture}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save, 'n' for next gesture",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show both original and mask
            cv2.imshow('Data Collection', frame)
            cv2.imshow('Hand Mask', mask)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and contour is not None:
                # Extract features and save
                features = self.extract_features_from_contour(contour)
                if len(features) > 0:
                    self.training_data.append(features)
                    self.training_labels.append(current_gesture)
                    current_samples += 1
                    print(f"Saved sample {current_samples} for {self.gesture_labels[current_gesture]}")

            elif key == ord('n'):
                # Move to next gesture
                current_gesture += 1
                current_samples = 0
                print(
                    f"Moving to next gesture: {self.gesture_labels[current_gesture] if current_gesture < len(self.gesture_labels) else 'Done'}")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save collected data
        if self.training_data:
            self.save_training_data()

    def save_training_data(self):
        data_dict = {
            'data': self.training_data,
            'labels': self.training_labels
        }

        with open('gesture_data_simple.pickle', 'wb') as f:
            pickle.dump(data_dict, f)
        print("Training data saved to gesture_data_simple.pickle")

    def load_training_data(self):
        try:
            with open('gesture_data_simple.pickle', 'rb') as f:
                data_dict = pickle.load(f)

            self.training_data = data_dict['data']
            self.training_labels = data_dict['labels']
            print(f"Loaded {len(self.training_data)} training samples")
            return True
        except FileNotFoundError:
            print("No training data found. Please collect data first.")
            return False

    def train_model(self):
        """Train the machine learning model"""
        if not self.training_data:
            print("No training data available!")
            return

        # Convert to numpy arrays
        X = np.array(self.training_data)
        y = np.array(self.training_labels)

        print(f"Training data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the model
        print("Training the model...")
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained with accuracy: {accuracy:.2f}")
        self.model_trained = True

        # Save the trained model
        with open('gesture_model_simple.pickle', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved to gesture_model_simple.pickle")

    def load_model(self):
        """Load pre-trained model"""
        try:
            with open('gesture_model_simple.pickle', 'rb') as f:
                self.model = pickle.load(f)
            self.model_trained = True
            print("Model loaded successfully")
            return True
        except FileNotFoundError:
            print("No trained model found. Please train the model first.")
            return False

    def real_time_recognition(self):
        """Perform real-time gesture recognition"""
        if not self.model_trained:
            print("Model not trained! Please train the model first.")
            return

        print("Starting real-time recognition...")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(0)

        # Learn background again
        print("Learning background for recognition...")
        for i in range(60):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                processed = self.preprocess_frame(frame)
                self.bg_subtractor.apply(processed)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Preprocess frame
            processed = self.preprocess_frame(frame)

            # Extract hand contour
            contour, mask = self.extract_hand_contour(processed)

            if contour is not None:
                # Draw contour
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                # Extract features and predict
                features = self.extract_features_from_contour(contour)
                if len(features) > 0:
                    try:
                        prediction = self.model.predict([features])[0]
                        confidence = np.max(self.model.predict_proba([features]))

                        # Display prediction
                        gesture_name = self.gesture_labels[prediction]
                        cv2.putText(frame, f"Gesture: {gesture_name}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except Exception as e:
                        cv2.putText(frame, "Processing...",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Gesture Recognition', frame)
            cv2.imshow('Hand Mask', mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the application"""
    recognizer = SimpleGestureRecognizer()

    while True:
        print("\n=== Simple Hand Gesture Recognition System ===")
        print("1. Collect Training Data")
        print("2. Train Model")
        print("3. Real-time Recognition")
        print("4. Load Existing Model")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            recognizer.collect_training_data()

        elif choice == '2':
            if recognizer.load_training_data():
                recognizer.train_model()

        elif choice == '3':
            if not recognizer.model_trained:
                recognizer.load_model()
            recognizer.real_time_recognition()

        elif choice == '4':
            recognizer.load_model()

        elif choice == '5':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()