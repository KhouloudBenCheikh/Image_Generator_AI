import tkinter as tk
import customtkinter as ctk
from authtoken import auth_token
from PIL import ImageTk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the Application Interface
app = tk.Tk()
app.geometry("600x600")
app.title("Image Generator AI")

# Filling the Background with an image 
#image_path = "Images/image.jpg"
#img = Image.open(image_path)
#bg_image = ImageTk.PhotoImage(img)
#bg_label = tk.Label(app, image=bg_image)
#bg_label.pack(fill=tk.BOTH, expand=True)


# Create an entry widget
prompt = ctk.CTkEntry(master=app, height=40, width= 560, text_color="black", fg_color="white")
prompt.pack()
prompt.place(x=20, y=30)

# Create a label widget
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

#Model
model_id = "CompVis/stable-diffusion-v1-4"
#For CPU users: device="cpu"
#For GPU users: device="cuda" 
device="cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    img = ImageTk.PhotoImage(image)
    img.save("Generated image.png")
    lmain.configure(image=img)
    
trigger = ctk.CTkButton(master=app,height=40, width= 90, text_color="white", fg_color="green", command=generate)
trigger.place(x=250, y=90)

app.mainloop()
