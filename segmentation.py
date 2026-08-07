from rembg import remove
from PIL import Image


def remove_bg(input_path, output_path="person_no_bg.png"):
    with open(input_path, "rb") as i:
        with open(output_path, "wb") as o:
            o.write(remove(i.read()))
    return output_path