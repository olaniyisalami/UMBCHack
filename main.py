import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import shutil
import os

def center_window(window):
    window.update_idletasks() 
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    window.geometry(f'{width}x{height}+{x}+{y}')
def load_image(file_path):
    try:
        # Open the selected image
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.Resampling.BOX)  # Resize image for display
        photo = ImageTk.PhotoImage(image)
        
        # Update the image label with the new image
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
    except Exception as e:
        print(f"Error loading image: {e}")

def ask_confirmation(classification_result):
    return messagebox.askyesno("Model Prediction", f"The model predicts that this is a {classification_result}. Is this correct?")
    
def ask_for_correct_classification():
    correct_window = tk.Toplevel(root)
    correct_window.title("Correct Classification")

    user_selection = tk.StringVar()

    label = tk.Label(correct_window, text="Please select the correct classification:")
    label.pack(pady=10)

    correct_combobox = ttk.Combobox(correct_window, values=class_names, textvariable=user_selection)
    correct_combobox.set("Select correct classification") 
    correct_combobox.pack(pady=10)

    confirm_button = tk.Button(correct_window, text="Confirm", command=lambda: confirm_classification(user_selection, correct_window))
    confirm_button.pack(pady=10)

    correct_window.wait_window()

    return user_selection.get()
def confirm_classification(selected_class, window):
    if selected_class in class_names:
        print(f"User selected the correct classification: {selected_class}")
    else:
        print("No valid classification selected.")
    window.destroy()

def choose_file():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.HEIC")]
    )
    if file_path:
        load_image(file_path)
        prediction = subprocess.check_output(["python", "modelTrain.py", file_path], text=True).strip()
        print(prediction)
        if ask_confirmation(prediction):
            try:
                source = file_path
                destination = 'trash/' + str(prediction) + '/' + str(os.path.basename(file_path))
                shutil.move(file_path, destination)
                print(f'File moved from {source} to {destination}')
            except FileNotFoundError as e:
                print(f'Error: {e}')
            except Exception as e:
                print(f'Unexpected error: {e}')
        else:
            try:
                correct = ask_for_correct_classification()
                source = file_path
                destination = 'trash/' + correct + '/' + str(os.path.basename(file_path))
                shutil.move(file_path, destination)
                print(f'File moved from {source} to {destination}')
            except FileNotFoundError as e:
                print(f'Error: {e}')
            except Exception as e:
                print(f'Unexpected error: {e}')

def run_main_train():
    subprocess.run(["python", "modelTrain.py"])
if not os.path.exists('./trash'):
    subprocess.run(["python", "modelTrain.py"])  
root = tk.Tk()
root.title("Image Classification GUI")

root.geometry("400x400")  # Set initial size (width x height)

# Center the window
center_window(root)
# Create a label to display the image (initially empty)
image_label = tk.Label(root)
image_label.pack(pady=20)  # Add some padding at the top

# Button to choose a file
choose_file_button = tk.Button(root, text="Choose Image", command=choose_file)
choose_file_button.pack(pady=10)

# Finished button to run main.py again
finished_button = tk.Button(root, text="Finished", command=lambda: run_main_train())
finished_button.pack(pady=10)

# Create a dropdown (combobox) for classification
class_names = ['Aluminium foil', 'Battery', 'Blister pack', 'Bottle', 'Bottle cap', 'Broken glass', 'Can', 'Carton', 'Cigarette', 'Cup', 'Food waste', 'Glass jar', 'Lid', 'Other plastic', 'Paper', 'Paper bag', 'Plastic bag & wrapper', 'Plastic container', 'Plastic glooves', 'Plastic utensils', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Squeezable tube', 'Straw', 'Styrofoam piece', 'Unlabeled litter']  # Replace with your class names
# Bind the select event

# Start the main event loop
root.mainloop()