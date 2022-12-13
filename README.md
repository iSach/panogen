# Background-Only Panorama Generation

In this project, we present an application for three tasks.

1. *Panorama Generation*: Creates a panorama of the background of the scene being filmed, removing persons and balls in the way.
2. *Ball tracking*: At any time, can track balls in the images, and give their 2D position on the image along with its distance from the camera.
3. *Motion Estimation*: A live estimation of the panning angle of the camera.

Along with that, the application allows for a lot of different features, that can be tweaked with parameters when running it. This includes exporting bounding boxes to a JSON file, displaying the angle, the bounding boxes and ball distances, changing the detection model to increase speed or accuracy, saving the video, etc.

This project is developed during the Computer Vision (ELEN0016) class at University of Liège, during Fall 2022.

The group members are, in alphabetical order:
- La Rocca Lionel
- Lewin Sacha
- Louette Arthur
- Maréchal Michaël
- Vinders Adrien

## Installation

One can create a virtual environment using the file `requirements.txt`:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running

To run the application, use:
```
python3 main.py [params]
```

Where the parameters are showed in the following table.

| Parameter   | Short name  | Description | Values |
| ----------- | ----------- | ----------- | ----------- |
| `--online` | `-o` | Capture mode, online? (Live) | bool (def. `True`) |
| `--path` | `-p` | Prefix path to sequence images | str (def. `data/img_5_1/img_5_1_`) |
| `--savevideo` | `-sv` | Save video? | bool (def. `True`) |
| `--displayvideo` | `-dv` | Display the video? | bool (def. `True`) |
| `--skip` | `-s` | Nb. of frames between two updates of the angle estimation | int (def. `0`) |
| `--print` | `-d` | Print the angle estimation in console? | bool (def. `False`) |
| `--padcount` | `-f` | Should pad frames ids? (0047 vs 47) | bool (def. `True`) |
| `--format` | `-ff` | Images format | str (def. `jpg`) |
| `--model` | `-m` | YOLO model to use for detection | str (def. `yolov5n`) |
| `--smalldiameter` | `-sd` | Small ball diameter in cm. | float (def. `10.5`) |
| `--bigdiameter` | `-bd` | Big ball diameter in cm. | float (def. `23.8`) |
| `--savejson` | `-sj` | Save Bounding Boxes to a JSON file? | bool (def. `True`) |
| `--panorama` | `-pan` | Generate a panorama? | bool (def. `False`) |
| `--showbbox` | `-sb` | Show bounding boxes on video? | bool (def. ``) |
| `--begin` | `-b` | First frame to start from | int (def. `0`) |
| `--maxframes` | `-mf` | Maximum number of frames (-1 for no limit) | int (def. `-1`) |

## Modules

For further documentation, see the comments in the files.

### `main.py`

Main file, used to run the application. Contains the loop that calls other files to merge every part of the application.

### `record.py`

Can be run in order to record a sequence as a list of jpg frames, at 1280x720, RGB, 25fps.
```
python3 record.py -n seq_name -m nb_frames
```

### `feed_reader.py`

Second acquisition file, dedicated to reading a video feed, either online directly from the camera, or offline from a sequence of files.

### `panorama.py`

Contains the code related to generation of the panorama using cylindrical warping.

### `json_writer.py`

Contains the code for exporting bounding boxes as a JSON file.

### `anglemeter.py`

Contains the code for the camera motion estimation part that evaluates the current angle of the camera compared to its resting (i.e. initial) position.

### `calibration.py`

Contains the code to calibrate the camera using a series of checkerboard images. Saves the camera matrix to a file after first calibration, in `data/calibration`

### `motion_detection.py`

Motion and object detection module, but also performs the segmentation masks. It uses YOLO to detect individual frames and return the bounding boxes and mask.

### `evaluation.ipynb`

Jupyter notebook for the evaluation of the different parts. Further documentation inside details the metrics used.

### `data` Folder

Contains sequences, calibration data, etc. A sequence should be saved as `data/seq_name/seq_name_{frame_nb}.jpg`.

### `references.md`

Contains references.

### `requirements.txt`

Contains dependencies.

### `make_release.sh`

Allows to build a zip archive for submission of the project.

## References

See file `references.md`.