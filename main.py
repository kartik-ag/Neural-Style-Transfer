import streamlit as st
from PIL import Image
import additional_files.style as style
import additional_files.load as load
import os
from io import BytesIO
import base64

# Style image paths:
root_style = "./images"

# Download image function
def get_image_download_link(img, file_name, style_name):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a style="color:black" href="data:file/jpg;base64,{img_str}" download="{style_name}_{file_name}.jpg"><input type="button" value="Download"></a>'
    return href

st.markdown("<h1 style='text-align: center;'>Neural Style Transfer</h1>", unsafe_allow_html=True)

# Creating a style selection in the main screen
style_name = st.selectbox(
    'Select Style',
    ("candy", "mosaic", "rain_princess", "udnie", "tg", "demon_slayer", "ben_giles", "ben_giles_2")
)
path_style = os.path.join(root_style, style_name + ".jpg")

# Upload image functionality
img = None
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

show_file = st.empty()

# Checking if user has uploaded any file
if not uploaded_file:
    show_file.info("Please Upload an Image")
else:
    # Size check for uploaded file (limit to 5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    if uploaded_file.size > MAX_FILE_SIZE:
        show_file.error("File size should not exceed 5MB.")
    else:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.image(path_style, caption='Style Image', use_column_width=True)

extensions = [".png", ".jpeg", ".jpg"]

if uploaded_file is not None and any(extension in uploaded_file.name for extension in extensions):
    name_file = uploaded_file.name.split(".")
    root_model = "./models"
    model_path = os.path.join(root_model, style_name + ".pth")

    if img is not None:
        img = img.convert('RGB')
        input_image = img

        root_output = "./images/output-images"
        output_image = os.path.join(root_output, style_name + "-" + name_file[0] + ".jpg")

        stylize_button = st.button("Stylize")

        if stylize_button:
            model = load.load_model(model_path)
            stylized = style.stylize(model, input_image, output_image)
            # Displaying the output image
            st.write("### Output Image")
            st.image(stylized, width=400, use_column_width=True)
            st.markdown(get_image_download_link(stylized, name_file[0], style_name), unsafe_allow_html=True)
