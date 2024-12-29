import customtkinter as ctk
import cv2
import os
import numpy as np
from PIL import Image
from tkinter import scrolledtext
import time
import shutil
from tkinter import simpledialog, messagebox
import webbrowser
import requests


# Initialize the main window
ctk.set_appearance_mode("Light")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

window = ctk.CTk()  # Using CTk instead of Tk
window.title("Face Recognition System")
window.geometry("600x500")  # Adjusted window size
window.resizable(0, 0)

###########################################################
# Create buttons using CustomTkinter
###########################################################

button_style = {
    'width': 250,  # Width of the button
    'height': 40,  # Height of the button
    'border_width': 2,  # Border width
    'corner_radius': 10,  # Rounded corners
    # 'fg_color': "#ffffff",  # Foreground (text) color
    # 'bg_color': "#00308F",  # Background color
    'hover_color': "#0076CE",  # Color when hovered over
    'text_color': "#ffffff",  # Text color
    'font': ("Poppins", 16),  # Font style and size
}

##############################################
#           Frame for user input
##############################################
input_frame = ctk.CTkFrame(window)
input_frame.pack(pady=20, padx=20, fill="x")

# Label and entry for Name input
name_label = ctk.CTkLabel(input_frame, text="Name", font=("Poppins", 16))
name_label.grid(row=0, column=0, padx=10, pady=10)

name_entry = ctk.CTkEntry(input_frame, width=410, font=("Helvetica", 14))  # CustomTkinter entry widget
name_entry.grid(row=0, column=1, padx=0, pady=10)

### Toggle Button for switching modes (dark n light)
def toggle_mode():
    current_mode = ctk.get_appearance_mode()
    if current_mode == "Light":
        ctk.set_appearance_mode("Dark")
    else:
        ctk.set_appearance_mode("Light")

# Create a toggle button for Dark/Light mode
toggle_button = ctk.CTkButton(input_frame, text="Theme", command=toggle_mode, width=40,text_color="#ffffff")
toggle_button.grid(row=0, column=2, padx=10, pady=10) 


##############################################
#           Status Area
##############################################
status_area = scrolledtext.ScrolledText(window, height=7, width=70, font=("Poppins", 12))
status_area.pack(pady=10)

# Configure tags for different colors
status_area.tag_config("success", foreground="green")
status_area.tag_config("error", foreground="red")
status_area.tag_config("info", foreground="black")
  
status_area.insert("end", "                                               ---Status updates will appear here---\n", "info")

# Disable the text area to make it read-only
status_area.config(state="disabled")

def log_status(message, tag=None):
    # Enable the text area to insert new meassage
    status_area.config(state="normal")
    if tag:
        status_area.insert("end", message + "\n", tag)
    else:
        status_area.insert("end", message + "\n") 
    status_area.yview("end")  # Scroll to the end
    status_area.update() # Ensure the GUI updates immediately

    # Disable it again to make it read-only
    status_area.config(state="disabled")


### Left Frame for Buttons
button_frame_left = ctk.CTkFrame(window)
button_frame_left.pack(padx=10, pady=10, side="left")

###########################################################
# Generate Dataset
###########################################################
def generate_dataset():
    data_dir = "dataset"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    name = name_entry.get().strip()
    if name == "":
        log_status('Please provide a name.', "error")
        return

    user_folder = os.path.join(data_dir, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    img_id = 0  # for naming the dataset

    log_status("Opening camera . . .", "info")  # Indicate camera opening
    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        log_status("Error: Could not open camera.", "error")
        return  # Exit if camera can't be opened
    else:
        log_status("Camera opened successfully.")  # Confirm successful opening

    while True:
        ret, frame = cap.read()
        if not ret:
            log_status("Failed to capture image.", "error")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5)

        for (x, y, w, h) in faces:
            cropped_face = frame[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (200, 200))

            cv2.imwrite(f"{user_folder}/{name}.{img_id}.jpg", cropped_face)
            img_id += 1

            cv2.putText(cropped_face, str(img_id), (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (180, 130, 70), 2)
            cv2.imshow("Cropped face", cropped_face)

        # Check for key press
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            log_status("Forcefully terminated, please capture again.", "error")
            break  # Exit the loop if Enter is pressed

        if img_id >= 100:
            log_status("Generating dataset completed! Please click on Train Classifier.", "success")
            break

    cap.release()
    cv2.destroyAllWindows()

# Button for generating dataset
b1 = ctk.CTkButton(button_frame_left, text="Step 1: Capture Face", command=generate_dataset, **button_style)
b1.pack(side="top", padx=10, pady=10)

###########################################################
# Training
###########################################################
def train_classifier():
    # Directory containing the dataset of face images
    data_dir = "dataset"
    
    log_status("Training started . . .", "info") #printing message

    faces = []  # List to store face images
    ids = []    # List to store corresponding IDs
    username_to_id = {}  # Mapping of usernames to IDs

    # Iterate through each person's directory in the dataset
    for current_id, person in enumerate(os.listdir(data_dir), start=1):
        person_dir = os.path.join(data_dir, person)

        # Continue if the path is not a directory
        if not os.path.isdir(person_dir):
            continue

        # Map the person's name to their ID
        username_to_id[person] = current_id

        # Iterate through each image in the person's directory
        for image in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image)

            # Read the image, convert to grayscale, and convert to a NumPy array
            img = Image.open(img_path).convert('L')
            imageNp = np.array(img, 'uint8')

            # Append the image and its corresponding ID
            faces.append(imageNp)
            ids.append(current_id)

    # Convert ids list to a NumPy array
    ids = np.array(ids)

    # Create and train the LBPH face recognizer
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Save the trained classifier and username-ID mapping
    clf.write("classifier.xml")
    np.save("username_to_id.npy", username_to_id)

    # Log completion status
    log_status('Training dataset completed', "success")

#Button for Training
b2 = ctk.CTkButton(button_frame_left, text="Step 2: Train Classifier", command=train_classifier, **button_style)
b2.pack(side="top", padx=10, pady=10)

###########################################################
# Detect Face
###########################################################
def detect_face():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    username_to_id = np.load("username_to_id.npy", allow_pickle=True).item()

    video_capture = cv2.VideoCapture(0)
    log_status("Camera opening. . .")
    time.sleep(0.5)
    log_status("Camera opened", "success")     


    while True:
        ret, img = video_capture.read()
        if not ret:
            log_status("Failed to capture image.", "error")
            break

        cv2.putText(img, "Press Enter to Close", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (180, 130, 70), 2, cv2.LINE_AA)  

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.05, 5)

        for (x, y, w, h) in faces:
            id, confidence = clf.predict(gray[y:y + h, x:x + w])
            confidence = int(100 * (1 - confidence / 400))

            if confidence > 77:
                name = [username for username, uid in username_to_id.items() if uid == id][0]
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX , 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_DUPLEX , 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1)& 0xFF == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

b3 = ctk.CTkButton(button_frame_left, text="Step 3: Detect Face", command=detect_face, **button_style)
b3.pack(side="top", padx=10, pady=10)

### Right Frame for Buttons
button_frame_right = ctk.CTkFrame(window)
button_frame_right.pack(padx=10, pady=10, side="right")

##############################################
#          Reset Dataset 
##############################################
def reset_dataset():
    # Show a confirmation dialog
    confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the dataset? This action cannot be undone.")
    
    if confirm:  # Proceed only if the user clicked "Yes"
        dataset_dir = "dataset"
        classifier_file = "classifier.xml"
        username_to_id_file = "username_to_id.npy"
        
        # Delete classifier.xml if it exists
        if os.path.exists(classifier_file):
            os.remove(classifier_file)
        
        # Delete username_to_id.npy if it exists
        if os.path.exists(username_to_id_file):
            os.remove(username_to_id_file)
        
        # Delete the dataset directory if it exists
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)  # Deletes the entire dataset folder
            log_status("Dataset and associated files reset successfully!", "success")
        else:
            log_status("Dataset not Found", "error")
    else:
        log_status("Dataset reset canceled", "info")  # Optional: Log if reset is canceled

# Button to reset dataset
b4 = ctk.CTkButton(button_frame_right, text="Reset Dataset", command=reset_dataset, **button_style)
b4.pack(side="top", padx=10, pady=10)

##############################################
#          Show Dataset 
##############################################
# Function to show usernames from the dataset folder
def show_dataset():
    dataset_dir = "dataset"

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        usernames = [username for username in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, username))]

        if usernames:
            # Ask the user to select a username to delete
            selected_user = simpledialog.askstring("Select User", "Enter username to delete:\n" + "\n".join(usernames))

            if selected_user in usernames:
                # Confirm deletion
                confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the user '{selected_user}'?")
                if confirm:
                    user_dir = os.path.join(dataset_dir, selected_user)
                    shutil.rmtree(user_dir)  # Delete the user's folder
                    messagebox.showinfo("Success", f"User '{selected_user}' deleted successfully!")
            else:
                messagebox.showerror("Error", "Username not found.")
        else:
            messagebox.showinfo("Usernames in Dataset", "No users found in the dataset.")
    else:
        messagebox.showerror("Error", "No dataset found")

# Button for showing the dataset
b5 = ctk.CTkButton(button_frame_right, text="Show/Delete Users", command=show_dataset, **button_style)
b5.pack(side="top", padx=10, pady=10)

##############################################
#          About Me
##############################################
def is_connected():
    try:
        # Check if we can reach Google's DNS server
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def open_browser():
    url = "https://www.linkedin.com/in/shivanshu-sawan/"  # Replace with your LinkedIn URL
    if is_connected():
        webbrowser.open(url)
    else:
        messagebox.showerror("Error", "No internet connection.")

b6 = ctk.CTkButton(button_frame_right, text="About Me", command=open_browser, **button_style)
b6.pack(side="top", padx=10, pady=10)


############################################################
############################################################
# Close window function
def close_window():
    window.destroy()

# Bind the cross button to the close function
window.protocol("WM_DELETE_WINDOW", close_window)

window.mainloop()