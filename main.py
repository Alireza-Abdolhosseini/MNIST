from torch import load as tload
from torchvision.io import read_image, ImageReadMode
from torch import argmax as targmax
from torch import max as tmax
from Neural_Network import Net
from torchvision.transforms import CenterCrop, Resize
import streamlit as st

picture = st.file_uploader("Choose a picture of a handwritten digit:", type=['png', 'jpg'])

if picture:
    with open("uploaded.jpg", "wb") as f:
        f.write(picture.read())


    img = read_image("uploaded.jpg", ImageReadMode.GRAY)
    c, h, w = img.shape

    if h <= w:
        img = CenterCrop(int(h))(img)
        st.image(img.reshape(h, h, 1).numpy(), width=300)
    else:
        img = CenterCrop(int(w))(img)
        st.image(img.reshape(w, w, 1).numpy(), width=300)

    img = Resize(size=(28, 28))(img)

    img = img / 255

    layers = [784, 300, 10]

    model = Net(layers)
    model.load_state_dict(tload("model.pt"))
    model.eval()


    # show_data(validation_dataset[1000][0])
    z = model(img.reshape(1, 1, 28, 28))
    a = int(targmax(z))
    if tmax(z) > 0.8:
        st.write(f"I am {int(round(float(tmax(z)), 2) * 100)}% sure this is number '{a}'.")
    else:
        st.write("I don't know! Are you sure this is a picture of a handwritten digit?")
