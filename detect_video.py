import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

from decord import VideoReader, cpu
import numpy as np
import jsonlines
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source="",  # file/dir/URL/glob/screen/0(webcam)
    save_dir="",
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    # 参数初始化
    img_size = 640
    stride = 32
    auto = True

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # json = {video_namme, frames, h, w, fps, use, frames_flag = {idx, det}}
    # source = []
    # if len(source) == 0:
    #     source += ["/home/ubuntu/user_space/zhangdy/shanhai-ai/dataset/bilibili_music/vid_2_512_25fps/210752/210752-20230328-BV1Sv4y1G7BM-1920x1080-4134-5089-0-625_512_25fps.mp4"]
    samples = list(csv.DictReader(open(source, "r", encoding="utf-8-sig")))#[::110] # 8582
    # samples = samples[:2100] # gpu0
    # out_jsonl_path = os.path.join(save_dir, 'lips_gpu0.jsonl')

    # samples = samples[2100:4200] # gpu1
    # out_jsonl_path = os.path.join(save_dir, 'lips_gpu1.jsonl')

    # samples = samples[4200:6300] # gpu2
    # out_jsonl_path = os.path.join(save_dir, 'lips_gpu2.jsonl')

    samples = samples[6300:] # gpu3
    out_jsonl_path = os.path.join(save_dir, 'lips_gpu03.jsonl')

    # out_jsonl_path = os.path.join(save_dir, 'lips_gpu0.jsonl')
    with jsonlines.open(out_jsonl_path, mode="w") as file_jsonl:
        for vid_path in samples:
            vid_path = vid_path["oss_url"]
            # print('111', vid_path)
            video_reader = VideoReader(vid_path, ctx=cpu(0))
            p = Path(vid_path)  # to Path
            height, width = video_reader[0].shape[:2]
            fps = video_reader.get_avg_fps()

            video_name = os.path.basename(vid_path).split('.')[0]
            temp_dict = dict(video_name=video_name, nframes=len(video_reader), H=height, W=width, fps=fps, use=1)
            # print(temp_dict)
            # exit()

            imgs = []
            for i in range(len(video_reader)):
                # the video reader will handle seeking and skipping in the most efficient manner
                imgs.append(video_reader[i].asnumpy())

            im = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0] for x in imgs])  # resize
            im = im.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            ims = np.ascontiguousarray(im)  # contiguous


            # Directories path_parent_path = path_str.parent
            base_dir_split = vid_path.split(video_name)[0].split('/')
            # print(base_dir_split)
            save_dir_ = os.path.join(save_dir, base_dir_split[-3], base_dir_split[-2])  # increment run
            os.makedirs(save_dir_, exist_ok=True)
            
            # Define the path for the CSV file
            csv_path = os.path.join(save_dir_, f"{video_name}.csv")
            # Dataloader
            bs = 1  # batch_size

            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
            count = 0
            nf = 1

            for im in ims:
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    if model.xml and im.shape[0] > 1:
                        ims = torch.chunk(im, im.shape[0], 0)

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(vid_path).stem, mkdir=True) if visualize else False
                    if model.xml and im.shape[0] > 1:
                        pred = None
                        for image in ims:
                            if pred is None:
                                pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                            else:
                                pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                        pred = [pred, None]
                    else:
                        pred = model(im, augment=augment, visualize=visualize)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Create or append to the CSV file
                def write_to_csv(image_name, frames_idx, prediction):
                    """Writes prediction data for an image to a CSV file, appending if the file exists."""
                    data = {"Image Name": image_name, "FramesIdx": frames_idx, "Prediction": prediction}
                    file_exists = os.path.isfile(csv_path)
                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=data.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(data)
                s = f"video: ({count+1}/{len(ims)}) {p.name}: " # count=video数量
                
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], imgs[0].shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        label = names[int(c)]
                    else:
                        temp_dict["use"]=0
                        label = 0
                
                write_to_csv(p.name, count, label)
                count+=1
                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            file_jsonl.write(temp_dict)
    file_jsonl.close()
def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--save_dir", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

# python detect.py --source E:/workdir/dataset/douyin_recoed/555.png --weights runs/weights/best.pt

# CUDA_VISIBLE_DEVICES=0 python detect_video.py --source DouyinTalk_video_oss_url_json250108.csv --save_dir ./runs --weights runs/train/exp2/weights/best.pt
# CUDA_VISIBLE_DEVICES=1 python detect_video.py --source DouyinTalk_video_oss_url_json250108.csv --save_dir ./runs --weights runs/train/exp2/weights/best.pt
# CUDA_VISIBLE_DEVICES=2 python detect_video.py --source DouyinTalk_video_oss_url_json250108.csv --save_dir ./runs --weights runs/train/exp2/weights/best.pt
# CUDA_VISIBLE_DEVICES=3 python detect_video.py --source DouyinTalk_video_oss_url_json250108.csv --save_dir ./runs --weights runs/train/exp2/weights/best.pt
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
