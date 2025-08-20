# Import libraries
import tkinter as tk
from tkinter import Canvas, Button, Label, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("model.h5")

# Define the GUI class
class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Digit Predictor")

        # Create the canvas to draw on
        self.canvas_width = 560
        self.canvas_height = 560
        self.canvas = Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(pady=10)

        # Create a PIL Image object to draw on
        self.img = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.img)

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=10)

        # Create buttons
        self.clear_button = Button(self.button_frame, text="Clear", command=self.clear_canvas, width=10)
        self.clear_button.grid(row=0, column=0, padx=10)

        self.predict_button = Button(self.button_frame, text="Predict", command=self.predict_digit, width=10)
        self.predict_button.grid(row=0, column=1, padx=10)

        self.help_button = Button(self.button_frame, text="Help", command=self.show_help, width=10)
        self.help_button.grid(row=0, column=2, padx=10)

        self.quit_button = Button(self.button_frame, text="Quit", command=master.quit, width=10)
        self.quit_button.grid(row=0, column=3, padx=10)

        # Create a label to display the predicted digit
        self.prediction_label = Label(master, text="", font=('Helvetica', 16))
        self.prediction_label.pack(pady=10)

    # Draw on the canvas when the mouse is dragged
    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill=0)

    # Clear the canvas
    def clear_canvas(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.prediction_label.config(text="")

    # Predict the digit
    def predict_digit(self):
        # Resize the image to 28x28, invert the colors, and smooth slightly
        img_resized = self.img.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)
        img_smoothed = img_inverted.filter(ImageFilter.SMOOTH_MORE)

        # Convert the image to a numpy array
        img_array = np.array(img_smoothed)
        img_array = img_array.astype("float32") / 255.0
        img_array = img_array.reshape((1, 28, 28, 1))

        # Make the prediction
        pred = model.predict(img_array)[0]
        digit = np.argmax(pred)
        accuracy = round(pred[digit]*100, 2)

        # Update the prediction label
        self.prediction_label.config(text=f"Predicted Digit: {digit} with {accuracy}% confidence")

    # Show the help information
    def show_help(self):
        help_text = (
            "Instructions:\n"
            "- Draw a digit (0-9) on the canvas.\n"
            "- Click 'Predict' to get the prediction.\n"
            "- Use 'Clear' to reset the canvas.\n"
            "- 'Quit' to exit the application."
        )
        messagebox.showinfo("Help", help_text)


# Create the main window and start the GUI
root = tk.Tk()
root.geometry("700x750")
root.resizable(False, False)  # Prevent window resizing
gui = GUI(root)
root.mainloop()
