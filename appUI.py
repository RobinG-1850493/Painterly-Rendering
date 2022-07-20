import PySimpleGUI as sg
from HertzmanRenderer import HertzmanRenderer
from HertzmanSimpleRenderer import HertzmanSimpleRenderer
from HertzmanVideoRenderer import HertzmanVideoRenderer
from io import BytesIO
import tkinter
import time
import threading
import cv2
import copy
import base64
import numpy as np
import PIL.Image, PIL.ImageTk



canvas_list = []
renderer = None
current_display = []

def setupGUI():
    sg.theme('DarkAmber')
    theme = "original"

    canvas_layout = [[sg.Text("Algorithm: "), sg.Combo(values = ["Hertzman Renderer", "Hertzman Video Renderer", "Simple Hertzman Renderer", "Particle"], enable_events = True, default_value = "Hertzman Renderer", key = "algorithm", size = 25, readonly = True)],
                [sg.Slider(orientation = "vertical", size = (29, 25), range = (3, 1), key = "layer_slider", disable_number_display = True, enable_events = True),
                 sg.Graph(background_color = "gray",canvas_size = (700, 600), graph_top_right = (700,600), graph_bottom_left = (0, 0),
                 p = 5, key = "-main_canvas-", expand_x  = True, expand_y = True, visible = True, border_width = 4)],
                [sg.Slider(orientation = "h", size = (100, 25), disable_number_display = True, enable_events = True, key = "comparison_slider", range = (0, 100), resolution = 5)]]

    settings_layout = [[sg.Text("Styles:"), sg.Combo(values = ["Custom", "Impressionist",  "Expressionist", "Colorist Wash", "Pointillist"], enable_events = True, default_value = "Custom", key = "style"),
                        sg.Text("Color Palette:"), sg.Combo(values=["Original", "Aquamarine", "Blue Tints", "Dark Greens", "Pastel", "Deep Purple", "Vibrant Purple", "Black Red Yellow", "Wide Range"], enable_events=True, default_value="Original", key="theme")],
                        [sg.Input(readonly = True, key = "path", text_color = "black"), sg.FileBrowse(file_types = (("*.png *", "*.jpg *", "*.raw *")))], 
                        [sg.Text("Parameters:")],
                        [sg.Text("Brush Sizes (3 sizes, separated by comma):"), sg.Input(size = 25, default_text = "8", key="brush_sizes")], [sg.Text("Blur Factor"), sg.Input(size = 25, default_text = "2", key = "blur")],
                        [sg.Text("Curvature Factor"), sg.Input(size=25, default_text="1", key="Curvature")],
                        [sg.Text("Error Threshold"), sg.Input(size = 25, default_text = "100", key = "threshold")], [sg.Text("Grid Size"), sg.Input(size = 25, default_text = "1", key = "grid_size")],
                        [sg.Text(text = "Minimum Stroke Length", key="min_stroke_label"), sg.Input(size = 25, default_text = "4", key = "min_stroke")], [sg.Text(text="Maximum Stroke Length",key="max_stroke_label"), sg.Input(size = 25, default_text = "16", key = "max_stroke")],
                        [sg.Text("R G B Jitter factors"), sg.Input(size=8, default_text="0", key="r_jitter"), sg.Input(size=8, default_text="0", key="g_jitter"), sg.Input(size=8, default_text="0", key="b_jitter")],
                        [sg.Text("H S V Jitter factors"), sg.Input(size=8, default_text="0", key="h_jitter"), sg.Input(size=8, default_text="0", key="s_jitter"), sg.Input(size=8, default_text="0", key="v_jitter")],
                        [sg.Button(button_text = "Render", size = (25, 3), pad = 5, bind_return_key = True, border_width = 1, enable_events = True)],
                        #[sg.Input(size=(25,200), expand_x = True, use_readonly_for_disable = True, border_width = 2.5, pad = 5, default_text = "Select an Image file (.png/.jpg) to begin.", key="progress_log")],
                        [sg.Text(text = "Select an Image file to begin.", size=(25, 12), expand_x = True, expand_y = True, border_width = 2.5, pad = 5, key="progress_log", background_color = "gray")], 
                        [sg.ProgressBar(size = (25, 25), k = "progress", orientation = "h", max_value = 100, expand_x = True, expand_y = True)],
                        [sg.Input(readonly = True, expand_x = True, expand_y = True, enable_events = True, key="save_path", text_color="black"), sg.SaveAs(button_text = "Save Image", key="save_file", file_types = (("*.png *",)))]]

    final_layout = [[sg.Column(canvas_layout), sg.Column(settings_layout, vertical_alignment = 'top', pad = (5,30))]]

    window = sg.Window('Painterly Rendering', final_layout)

    while True:
        event, values = window.read()
        print(event)
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        elif event == "style":
            # radi, threshold, curvature, grid_size, blur,  min_strokes, max_strokes, r_j, g_j, b_j, h_j, s_j, v_j, opacity
            """impressionist = [(2,4,8), 100, 1, 1, 2, 4, 16, (0, 0, 0), (0, 0, 0), 1]
            expressionist = [(2,4,8), 50, 0.25, 1, 0.5, 10, 16, (0, 0, 0), (0, 0, 0.5), 0.7]
            colorist_wash = [(2,4,8), 200, 1, 1, 0.5, 4, 16, (0.3, 0.3, 0.3), (0, 0, 0), 0.5]
            pointilist = [(2,4), 100, 1, 0.5, 0.5, 0, 0, (0, 0, 0), (0.3, 0, 1), 1]"""

            if values["style"] == "Impressionist":
                window["brush_sizes"].update(value="2,4,8")
                window["threshold"].update(value="100")
                window["Curvature"].update(value="1")
                window["grid_size"].update(value="1")
                window["blur"].update(value="2")
                window["min_stroke"].update(value="4")
                window["max_stroke"].update(value="16")
                window["r_jitter"].update(value="0")
                window["g_jitter"].update(value="0")
                window["b_jitter"].update(value="0")
                window["h_jitter"].update(value="0")
                window["s_jitter"].update(value="0")
                window["v_jitter"].update(value="0")
            elif values["style"] == "Expressionist":
                window["brush_sizes"].update(value="2,4,8")
                window["threshold"].update(value="50")
                window["Curvature"].update(value="0.25")
                window["grid_size"].update(value="1")
                window["blur"].update(value="4")
                window["min_stroke"].update(value="10")
                window["max_stroke"].update(value="16")
                window["r_jitter"].update(value="0")
                window["g_jitter"].update(value="0")
                window["b_jitter"].update(value="0")
                window["h_jitter"].update(value="0")
                window["s_jitter"].update(value="0")
                window["v_jitter"].update(value="0.5")
            elif values["style"] == "Colorist Wash":
                window["brush_sizes"].update(value="2,4,8")
                window["threshold"].update(value="200")
                window["Curvature"].update(value="1")
                window["grid_size"].update(value="1")
                window["blur"].update(value="2")
                window["min_stroke"].update(value="4")
                window["max_stroke"].update(value="16")
                window["r_jitter"].update(value="0.22")
                window["g_jitter"].update(value="0.22")
                window["b_jitter"].update(value="0.22")
                window["h_jitter"].update(value="0")
                window["s_jitter"].update(value="0")
                window["v_jitter"].update(value="0")
            elif values["style"] == "Pointillist":
                window["brush_sizes"].update(value="2,4")
                window["threshold"].update(value="100")
                window["Curvature"].update(value="1")
                window["grid_size"].update(value="1")
                window["blur"].update(value="2")
                window["min_stroke"].update(value="0")
                window["max_stroke"].update(value="0")
                window["r_jitter"].update(value="0")
                window["g_jitter"].update(value="0")
                window["b_jitter"].update(value="0")
                window["h_jitter"].update(value="0.3")
                window["s_jitter"].update(value="0")
                window["v_jitter"].update(value="1")

        #["Original", "Aquamarine", "Blue Tints", "Dark Greens", "Pastel", "Deep Purple", "Vibrant Purple", "Black Red Yellow", "Wide Range"]
        elif event == "theme":
            if values['theme'] == "Original":
                theme = "original"
            elif values['theme'] == "Aquamarine":
                theme = "hue_palettes/aquamarine.png"
            elif values['theme'] == "Blue Tints":
                theme = "hue_palettes/blue_hues.png"
            elif values['theme'] == "Dark Greens":
                theme = "hue_palettes/dark_greens.png"
            elif values['theme'] == "Pastel":
                theme = "hue_palettes/pastel.png"
            elif values['theme'] == "Deep Purple":
                theme = "hue_palettes/deep_purple.png"
            elif values['theme'] == "Vibrant Purple":
                theme = "hue_palettes/vibrant_purple.png"
            elif values['theme'] == "Black Red Yellow":
                theme = "hue_palettes/vibrant_red_yellow.png"
            elif values['theme'] == "Wide Range":
                theme = "hue_palettes/wide_colors.png"


        elif event == "Render":
            if values["path"] == "":
                current = window["progress_log"].get()
                window["progress_log"].update(current + "\nSelecting Image is required.")

            elif values["algorithm"] == "Hertzman Renderer":
                prepHertzman(values, window, False, theme)

            elif values["algorithm"] == "Hertzman Video Renderer":
                prepHertzman(values, window, True, theme)

            elif values["algorithm"] == "Simple Hertzman Renderer":
                prepSimpleHerzman(values, window)
                
        elif event == "layer_slider":
            if canvas_list != []:
                layer = int(values["layer_slider"]) - 1

                if values["comparison_slider"] != 0:
                    createComparison(renderer.getOriginal(), canvas_list[layer], window, values["comparison_slider"])
                else:
                    displayImage(canvas_list[layer], window)

        elif event == "comparison_slider":
            if canvas_list != [] and renderer != None:
                original = renderer.getOriginal()
                layer = int(values["layer_slider"]) - 1
                percentage = values["comparison_slider"]
                createComparison(original, canvas_list[layer], window, percentage)

        elif event == "save_path":
            print("Saving")
            if current_display != []:
                print("Hi")
                saveImage(values["save_path"])

        elif event == "algorithm":
            if values["algorithm"] == "Hertzman Renderer":
                window["min_stroke"].update(visible=True)
                window["min_stroke_label"].update(visible=True)
                window["max_stroke"].update(visible=True)
                window["max_stroke_label"].update(visible=True)

            elif values["algorithm"] == "Simple Hertzman Renderer":
                window["min_stroke"].update(visible=False)
                window["min_stroke_label"].update(visible=False)
                window["max_stroke"].update(visible=False)
                window["max_stroke_label"].update(visible=False)
                
                       
def prepSimpleHerzman(values, window):
    print("Starting Rendering")
    path = values["path"]
    temp = values["brush_sizes"].split(",")

    brush_sizes = []

    for val in temp:
        brush_sizes.append(int(val))

    brush_sizes = tuple(brush_sizes)
    window["layer_slider"].update(value = len(brush_sizes), range = (len(brush_sizes) ,1))

    blur_grade = float(values["blur"])
    threshold = int(values["threshold"])
    grid_size = float(values["grid_size"])

    layer = int(values["layer_slider"]) - 1
    
    x = threading.Thread(target=runSimpleHertzman, args=(path, brush_sizes, blur_grade, threshold, grid_size, window, layer))
    x.start()

def runSimpleHertzman(path, brush_sizes, blur_grade, threshold, grid_size, window, layer):
    global canvas_list, renderer

    renderer = HertzmanSimpleRenderer(path, brush_sizes, blur_grade, threshold, grid_size, window)
    canvas_list = renderer.paint()

    layer = len(brush_sizes) - 1

    displayImage(canvas_list[layer], window)



def prepHertzman(values, window, video, theme):
    print("Starting Rendering")
    path = values["path"]
    temp = values["brush_sizes"].split(",")

    brush_sizes = []

    for val in temp:
        brush_sizes.append(int(val))

    brush_sizes = tuple(brush_sizes)
    window["layer_slider"].update(value = len(brush_sizes), range = (len(brush_sizes) ,1))

    blur_grade = float(values["blur"])
    threshold = int(values["threshold"])
    grid_size = float(values["grid_size"])
    min_stroke = int(values["min_stroke"])
    max_stroke = int(values["max_stroke"])

    rgb_j = (float(values["b_jitter"]), float(values["g_jitter"]), float(values["r_jitter"]))
    hsv_j = (float(values["h_jitter"]), float(values["s_jitter"]), float(values["v_jitter"]))

    curvature = float(values["Curvature"])

    layer = int(values["layer_slider"]) - 1

    #runHertzman(path, brush_sizes, blur_grade, threshold, grid_size, min_stroke, max_stroke, window, layer)
    
    print(video)
    if video:
        #runVideoHertzman(path, brush_sizes, blur_grade, threshold, curvature, grid_size, min_stroke, max_stroke, window, layer, rgb_j, hsv_j)
        x = threading.Thread(target=runVideoHertzman, args=(path, brush_sizes, blur_grade, threshold, curvature, grid_size, min_stroke, max_stroke, window, layer, rgb_j, hsv_j))
        x.start()
    else:
        x = threading.Thread(target=runHertzman, args=(path, brush_sizes, blur_grade, threshold, curvature, grid_size, min_stroke, max_stroke, window, layer, rgb_j, hsv_j, theme))
        x.start()


def runHertzman(path, brush_sizes, blur_grade, threshold, curvature, grid_size, min_stroke, max_stroke, window, layer, rgb_j, hsv_j, theme):
    global canvas_list, renderer
    
    renderer = HertzmanRenderer(path, brush_sizes, blur_grade, threshold, curvature, grid_size, min_stroke, max_stroke, window, rgb_j, hsv_j, theme)
    canvas_list = renderer.paint()

    start = time.time()
    displayImage(canvas_list[layer], window)
    end = time.time()
    print("Display Time spent: ",end - start)


def runVideoHertzman(path, brush_sizes, blur_grade, threshold, curvature, grid_size, min_stroke, max_stroke, window, layer, rgb_j, hsv_j):
    global canvas_list, renderer
    
    renderer = HertzmanVideoRenderer(path, brush_sizes, blur_grade, threshold, curvature, grid_size, min_stroke, max_stroke, window, rgb_j, hsv_j)
    video_canvas = renderer.paint()

    displayVideo(video_canvas, window)


def saveImage(path):
    print(path[:len(path)-4])
    if path[len(path)-4:] != ".png":
        path = path + ".png"

    converted_image = PIL.Image.fromarray(current_display[:,:,::-1])
    converted_image.save(path, format='PNG')

    
def displayImage(canvas, window):
    global current_display

    graph = window["-main_canvas-"]

    canvas = canvas * 255
    canvas = canvas.astype("uint8")
    #canvas = canvas[:,:,::-1]

    current_display = copy.copy(canvas)
    canvas = cv2.resize(canvas, (graph.get_size()[0], graph.get_size()[1]), interpolation = cv2.INTER_NEAREST)

    _, im_arr = cv2.imencode('.png', canvas)
    img_str = im_arr.tobytes()


    """converted_image = PIL.Image.fromarray(canvas)
    current_display = converted_image    

    converted_image = converted_image.resize((graph.get_size()[0], graph.get_size()[1]), PIL.Image.ANTIALIAS)

    buffered = BytesIO()
    converted_image.save(buffered, format="PNG")
    img_str = buffered.getvalue()"""

    #img_str = cv2.imencode('.png', canvas)[1].tobytes()   

    graph.draw_image(data=img_str, location=(0, 600))

def displayVideo(video, window):
    return 0

def createComparison(original, canvas, window, percentage):
    width = original.shape[1]
    original_share = int(width * (percentage / 100))
    
    original.astype("float")
    original = original / 255
    original = original[:, :original_share, :]
    canvas = canvas[:, original_share:, :]

    new_image = np.concatenate((original, canvas), axis=1)

    displayImage(new_image, window)


if __name__ == "__main__":
    setupGUI()
