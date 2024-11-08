import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import folder_paths
import comfy.utils
import time
import copy
import dill, json
import yaml
from ultralytics import YOLO

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

from .LivePortrait.live_portrait_wrapper import LivePortraitWrapper
from .LivePortrait.utils.camera import get_rotation_matrix
from .LivePortrait.config.inference_config import InferenceConfig

from .LivePortrait.modules.spade_generator import SPADEDecoder
from .LivePortrait.modules.warping_network import WarpingNetwork
from .LivePortrait.modules.motion_extractor import MotionExtractor
from .LivePortrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from .LivePortrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork
from collections import OrderedDict

cur_device = None
def get_device():
    global cur_device
    if cur_device == None:
        if torch.cuda.is_available():
            cur_device = torch.device('cuda')
            print("Uses CUDA device.")
        elif torch.backends.mps.is_available():
            cur_device = torch.device('mps')
            print("Uses MPS device.")
        else:
            cur_device = torch.device('cpu')
            print("Uses CPU device.")
    return cur_device

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def rgb_crop(rgb, region):
    return rgb[region[1]:region[3], region[0]:region[2]]

def rgb_crop_batch(rgbs, region):
    return rgbs[:, region[1]:region[3], region[0]:region[2]]
def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]
def create_transform_matrix(x, y, s_x, s_y):
    return np.float32([[s_x, 0, x], [0, s_y, y]])

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

def calc_crop_limit(center, img_size, crop_size):
    pos = center - crop_size / 2
    if pos < 0:
        crop_size += pos * 2
        pos = 0

    pos2 = pos + crop_size

    if img_size < pos2:
        crop_size -= (pos2 - img_size) * 2
        pos2 = img_size
        pos = pos2 - crop_size

    return pos, pos2, crop_size

def retargeting(delta_out, driving_exp, factor, idxes):
    for idx in idxes:
        #delta_out[0, idx] -= src_exp[0, idx] * factor
        delta_out[0, idx] += driving_exp[0, idx] * factor

class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori

import requests
from tqdm import tqdm

class LP_Engine:
    pipeline = None
    detect_model = None
    mask_img = None
    temp_img_idx = 0

    def get_temp_img_name(self):
        self.temp_img_idx += 1
        return "expression_edit_preview" + str(self.temp_img_idx) + ".png"

    def download_model(_, file_path, model_url):
        print('AdvancedLivePortrait: Downloading model...')
        response = requests.get(model_url, stream=True)
        try:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                # tqdm will display a progress bar
                with open(file_path, 'wb') as file, tqdm(
                        desc='Downloading',
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        file.write(data)

        except requests.exceptions.RequestException as err:
            print('AdvancedLivePortrait: Model download failed: {err}')
            print(f'AdvancedLivePortrait: Download it manually from: {model_url}')
            print(f'AdvancedLivePortrait: And put it in {file_path}')
        except Exception as e:
            print(f'AdvancedLivePortrait: An unexpected error occurred: {e}')

    def remove_ddp_dumplicate_key(_, state_dict):
        state_dict_new = OrderedDict()
        for key in state_dict.keys():
            state_dict_new[key.replace('module.', '')] = state_dict[key]
        return state_dict_new

    def filter_for_model(_, checkpoint, prefix):
        filtered_checkpoint = {key.replace(prefix + "_module.", ""): value for key, value in checkpoint.items() if
                               key.startswith(prefix)}
        return filtered_checkpoint

    def load_model(self, model_config, model_type):

        device = get_device()

        if model_type == 'stitching_retargeting_module':
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "retargeting_models", model_type + ".pth")
        else:
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "base_models", model_type + ".pth")

        is_safetensors = None
        if os.path.isfile(ckpt_path) == False:
            is_safetensors = True
            ckpt_path = os.path.join(get_model_dir("liveportrait"), model_type + ".safetensors")
            if os.path.isfile(ckpt_path) == False:
                self.download_model(ckpt_path,
                "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/" + model_type + ".safetensors")
        model_params = model_config['model_params'][f'{model_type}_params']
        if model_type == 'appearance_feature_extractor':
            model = AppearanceFeatureExtractor(**model_params).to(device)
        elif model_type == 'motion_extractor':
            model = MotionExtractor(**model_params).to(device)
        elif model_type == 'warping_module':
            model = WarpingNetwork(**model_params).to(device)
        elif model_type == 'spade_generator':
            model = SPADEDecoder(**model_params).to(device)
        elif model_type == 'stitching_retargeting_module':
            # Special handling for stitching and retargeting module
            config = model_config['model_params']['stitching_retargeting_module_params']
            checkpoint = comfy.utils.load_torch_file(ckpt_path)

            stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
            if is_safetensors:
                stitcher.load_state_dict(self.filter_for_model(checkpoint, 'retarget_shoulder'))
            else:
                stitcher.load_state_dict(self.remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
            stitcher = stitcher.to(device)
            stitcher.eval()

            return {
                'stitching': stitcher,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


        model.load_state_dict(comfy.utils.load_torch_file(ckpt_path))
        model.eval()
        return model

    def load_models(self):
        model_path = get_model_dir("liveportrait")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model_config_path = os.path.join(current_directory, 'LivePortrait', 'config', 'models.yaml')
        model_config = yaml.safe_load(open(model_config_path, 'r'))

        appearance_feature_extractor = self.load_model(model_config, 'appearance_feature_extractor')
        motion_extractor = self.load_model(model_config, 'motion_extractor')
        warping_module = self.load_model(model_config, 'warping_module')
        spade_generator = self.load_model(model_config, 'spade_generator')
        stitching_retargeting_module = self.load_model(model_config, 'stitching_retargeting_module')

        self.pipeline = LivePortraitWrapper(InferenceConfig(), appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module)

    def get_detect_model(self):
        if self.detect_model == None:
            model_dir = get_model_dir("ultralytics")
            if not os.path.exists(model_dir): os.mkdir(model_dir)
            model_path = os.path.join(model_dir, "face_yolov8n.pt")
            if not os.path.exists(model_path):
                self.download_model(model_path, "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
            self.detect_model = YOLO(model_path)

        return self.detect_model

    def get_face_bboxes(self, image_rgb):
        detect_model = self.get_detect_model()
        pred = detect_model(image_rgb, conf=0.7, device="")
        return pred[0].boxes.xyxy.cpu().numpy()

    def detect_face(self, image_rgb, crop_factor, sort = True):
        bboxes = self.get_face_bboxes(image_rgb)
        w, h = get_rgb_size(image_rgb)

        print(f"w, h:{w, h}")

        cx = w / 2
        min_diff = w
        best_box = None
        for x1, y1, x2, y2 in bboxes:
            bbox_w = x2 - x1
            if bbox_w < 30: continue
            diff = abs(cx - (x1 + bbox_w / 2))
            if diff < min_diff:
                best_box = [x1, y1, x2, y2]
                print(f"diff, min_diff, best_box:{diff, min_diff, best_box}")
                min_diff = diff

        if best_box == None:
            print("Failed to detect face!!")
            return [0, 0, w, h]

        x1, y1, x2, y2 = best_box

        #for x1, y1, x2, y2 in bboxes:
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        crop_w = bbox_w * crop_factor
        crop_h = bbox_h * crop_factor

        crop_w = max(crop_h, crop_w)
        crop_h = crop_w

        kernel_x = int(x1 + bbox_w / 2)
        kernel_y = int(y1 + bbox_h / 2)

        new_x1 = int(kernel_x - crop_w / 2)
        new_x2 = int(kernel_x + crop_w / 2)
        new_y1 = int(kernel_y - crop_h / 2)
        new_y2 = int(kernel_y + crop_h / 2)

        if not sort:
            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        elif w < new_x2:
            new_x1 -= (new_x2 - w)
            new_x2 = w
            if new_x1 < 0:
                new_x2 -= new_x1
                new_x1 = 0

        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        elif h < new_y2:
            new_y1 -= (new_y2 - h)
            new_y2 = h
            if new_y1 < 0:
                new_y2 -= new_y1
                new_y1 = 0

        if w < new_x2 and h < new_y2:
            over_x = new_x2 - w
            over_y = new_y2 - h
            over_min = min(over_x, over_y)
            new_x2 -= over_min
            new_y2 -= over_min

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


    def calc_face_region(self, square, dsize):
        region = copy.deepcopy(square)
        is_changed = False
        if dsize[0] < region[2]:
            region[2] = dsize[0]
            is_changed = True
        if dsize[1] < region[3]:
            region[3] = dsize[1]
            is_changed = True

        return region, is_changed

    def expand_img(self, rgb_img, square):
        #new_img = rgb_crop(rgb_img, face_region)
        crop_trans_m = create_transform_matrix(max(-square[0], 0), max(-square[1], 0), 1, 1)
        new_img = cv2.warpAffine(rgb_img, crop_trans_m, (square[2] - square[0], square[3] - square[1]),
                                        cv2.INTER_LINEAR)
        return new_img

    def get_pipeline(self):
        if self.pipeline == None:
            print("Load pipeline...")
            self.load_models()

        return self.pipeline

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256,256]
        if h != input_shape[0] or w != input_shape[1]:
            if 256 < h: interpolation = cv2.INTER_AREA
            else: interpolation = cv2.INTER_LINEAR
            x = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation = interpolation)
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(get_device())
        return x

    def GetMaskImg(self):
        if self.mask_img is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./LivePortrait/utils/resources/mask_template.png")
            self.mask_img = cv2.imread(path, cv2.IMREAD_COLOR)
        return self.mask_img

    def crop_face(self, img_rgb, crop_factor):
        crop_region = self.detect_face(img_rgb, crop_factor)
        face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))
        face_img = rgb_crop(img_rgb, face_region)
        if is_changed: face_img = self.expand_img(face_img, crop_region)
        return face_img

    def prepare_source(self, source_image, crop_factor, is_video = False, tracking = False):
        print("Prepare source...")
        engine = self.get_pipeline()
        source_image_np = (source_image * 255).byte().numpy()
        img_rgb = source_image_np[0]

        psi_list = []
        for img_rgb in source_image_np:
            if tracking or len(psi_list) == 0:
                crop_region = self.detect_face(img_rgb, crop_factor)
                face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))

                s_x = (face_region[2] - face_region[0]) / 512.
                s_y = (face_region[3] - face_region[1]) / 512.
                crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s_x, s_y)
                mask_ori = cv2.warpAffine(self.GetMaskImg(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR)
                mask_ori = mask_ori.astype(np.float32) / 255.

                if is_changed:
                    s = (crop_region[2] - crop_region[0]) / 512.
                    crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s, s)

            face_img = rgb_crop(img_rgb, face_region)
            if is_changed: face_img = self.expand_img(face_img, crop_region)
            i_s = self.prepare_src_image(face_img)
            x_s_info = engine.get_kp_info(i_s)
            f_s_user = engine.extract_feature_3d(i_s)
            x_s_user = engine.transform_keypoint(x_s_info)
            psi = PreparedSrcImg(img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori)
            if is_video == False:
                return psi
            psi_list.append(psi)

        return psi_list

    def prepare_driving_video(self, face_images):
        print("Prepare driving video...")
        pipeline = self.get_pipeline()
        f_img_np = (face_images * 255).byte().numpy()

        out_list = []
        for f_img in f_img_np:
            i_d = self.prepare_src_image(f_img)
            d_info = pipeline.get_kp_info(i_d)
            out_list.append(d_info)

        return out_list

    def calc_fe(_, x_d_new, eyes, eyebrow, wink, pupil_x, pupil_y, mouth, eee, woo, smile,
                rotate_pitch, rotate_yaw, rotate_roll):

        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        #x_d_new[0, 19, 1] += eee * 0.0006
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003
        x_d_new[0, 1, 1] += eyes * -0.00025
        x_d_new[0, 2, 1] += eyes * 0.00025


        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003


        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])
g_engine = LP_Engine()

class ExpressionSet:
    def __init__(self, erst = None, es = None):
        if es != None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(get_device())
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0
    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value
    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t
    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t
    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value

    #def apply_ratio(self, ratio):        self.exp *= ratio
    def to_dict(self)-> dict:
        return {
            'exp': self.e.to('cpu').tolist(),
            'rotation': self.r.to('cpu').tolist(),
            'scale': self.s,
            't': self.t
        }

    def from_dict(self, dict_erst: dict):
        e_np = np.array(dict_erst['exp'], dtype=np.float32)
        r_np = np.array(dict_erst['rotation'], dtype=np.float32)
        
        self.e = torch.from_numpy(e_np).float().to(get_device())
        self.r = torch.from_numpy(r_np).float()
        self.s = dict_erst['scale']
        self.t = dict_erst['t']


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time - start_time))
        return result

    return wrapper_fn


#exp_data_dir = os.path.join(current_directory, "exp_data")
exp_data_dir = os.path.join(folder_paths.output_directory, "exp_data")
if os.path.isdir(exp_data_dir) == False:
    os.mkdir(exp_data_dir)

class SaveExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "file_name": ("STRING", {"multiline": False, "default": ""}),
                "file_format": (["json", "dill"],),
            },
            "optional": {"save_exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_name",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"
    OUTPUT_NODE = True

    def run(self, file_name, file_format="json", save_exp:ExpressionSet=None):
        if save_exp == None or file_name == "":
            return file_name

        if file_format == "json":
            with open(os.path.join(exp_data_dir, file_name + ".json"), "w") as f:
                json.dump(save_exp.to_dict(), f, indent=4)
        elif file_format == "dill":
            with open(os.path.join(exp_data_dir, file_name + ".exp"), "wb") as f:
                dill.dump(save_exp, f)

        return file_name

class LoadExpData:
    @classmethod
    def INPUT_TYPES(s):
        file_list = [os.path.splitext(file)[0] for file in os.listdir(exp_data_dir) if file.endswith('.exp')]
        return {"required": {
            "file_name": (sorted(file_list, key=str.lower),),
            "ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, file_name, ratio):
        with open(os.path.join(exp_data_dir, file_name + ".exp"), 'rb') as f:
            es = dill.load(f)
        es.mul(ratio)
        return (es,)

class LoadExpDataJson:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "file_path": ("STRING", {"default": '', "multiline": False}),
        },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def __init__(self) -> None:
        self.file_path = ''
        self.es = None

    def run(self, file_path):
        if self.file_path != file_path:

            with open(os.path.join(exp_data_dir, file_path), 'r') as f:
                file_data = json.load(f)

            self.es = ExpressionSet()
            self.es.from_dict(file_data)

            self.file_path = file_path

        return (self.es,)

class LoadExpDataString:
    @classmethod
    def INPUT_TYPES(s):
        file_list = [os.path.splitext(file)[0] for file in os.listdir(exp_data_dir) if file.endswith('.json')]
        return {"required": {
            "text": ("STRING", {"default": '', "multiline": True}),
            "ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, text, ratio):        
        text_data = json.loads(text)
        es = ExpressionSet()
        es.from_dict(text_data)

        es.mul(ratio)
        return (es,)

class ExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                #"code": ("STRING", {"multiline": False, "default": ""}),
                "code1": ("INT", {"default": 0}),
                "value1": ("FLOAT", {"default": 0, "min": -500, "max": 500, "step": 0.1}),
                "code2": ("INT", {"default": 0}),
                "value2": ("FLOAT", {"default": 0, "min": -500, "max": 500, "step": 0.1}),
                "code3": ("INT", {"default": 0}),
                "value3": ("FLOAT", {"default": 0, "min": -500, "max": 500, "step": 0.1}),
                "code4": ("INT", {"default": 0}),
                "value4": ("FLOAT", {"default": 0, "min": -500, "max": 500, "step": 0.1}),
                "code5": ("INT", {"default": 0}),
                "value5": ("FLOAT", {"default": 0, "min": -500, "max": 500, "step": 0.1}),
            },
            "optional":{"add_exp": ("EXP_DATA",),}
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, code1, value1, code2, value2, code3, value3, code4, value4, code5, value5, add_exp=None):
        if add_exp == None:
            es = ExpressionSet()
        else:
            es = ExpressionSet(es = add_exp)

        codes = [code1, code2, code3, code4, code5]
        values = [value1, value2, value3, value4, value5]
        for i in range(5):
            idx = int(codes[i] / 10)
            r = codes[i] % 10
            es.e[0, idx, r] += values[i] * 0.001

        return (es,)

NAMES_EXP_BONE = [
    '额头', # 0
    '左眉头', # 1
    '右眉头', # 2
    '左腮骨', # 3
    '右耳根', # 4
    'root', # 5
    '6', # 6
    '右腮骨', # 7
    '8', # 8
    '9', # 9
    '左耳根', # 10
    '左眼珠', # 11
    '12', # 12
    '左眼袋', # 13
    '右嘴角', # 14
    '右眼珠', # 15
    '右眼袋', # 16
    '上嘴唇', # 17
    '18', # 18
    '下嘴唇', # 19
    '左嘴角', # 20   
]

class ShowExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{  
                "exp": ("EXP_DATA",),
                "id1": ("INT", {"default": 1}),
                "id2": ("INT", {"default": 2}),
                "id3": ("INT", {"default": 11}),
                "id4": ("INT", {"default": 15}),
                "id5": ("INT", {"default": 13}),
                "id6": ("INT", {"default": 16}),
                "ndigits": ("INT", {"default": 5, "min": 1, "max": 10, "step":1 }),
            },
            "optional":{}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, exp, id1, id2, id3, id4, id5, id6, ndigits):
        es = ExpressionSet(es = exp)
        lines = []
        string_format = '\{3\}\\t\{0:.{0}f\}, \{1:.5f\}, \{2:.5f\}'

        ids = [id1, id2, id3, id4, id5, id6]
        for i in ids:
            text_line = '{3}\t{0:.5f}, {1:.5f}, {2:.5f}'.format(
                round(es.e[0, i, 0].item(), ndigits),
                round(es.e[0, i, 1].item(), ndigits),
                round(es.e[0, i, 2].item(), ndigits),
                NAMES_EXP_BONE[i]
            )
            lines.append(text_line)

        text = '\n'.join(lines)
        return (text,)

class EditExpData:
    @classmethod
    def INPUT_TYPES(s):
        step = 0.0001
        return {"required":{                
                "id1": ("INT", {"default": 0, "min": -1, "max": 20}),
                "x1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "y1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "z1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "id2": ("INT", {"default": 0, "min": -1, "max": 20}),
                "x2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "y2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "z2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
            },
            "optional":{"exp": ("EXP_DATA",),}
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    @staticmethod
    def edit_idx(exp: ExpressionSet, idx: int, x, y, z):
        if idx == -1:
            exp.r[0] += x
            exp.r[1] += y
            exp.r[2] += z
        else:
            exp.e[0, idx, 0] = x
            exp.e[0, idx, 1] = y
            exp.e[0, idx, 2] = z

    def run(self, id1, x1, y1, z1, id2, x2, y2, z2, exp=None):
        if exp is None:
            es = ExpressionSet()
        else:
            es = ExpressionSet(es = exp)

        self.edit_idx(es, id1, x1, y1, z1)
        self.edit_idx(es, id2, x2, y2, z2)

        return (es,)

class EditExpDataRough(EditExpData):
    @classmethod
    def INPUT_TYPES(s):
        step = 0.01
        return {"required":{                
                "id1": ("INT", {"default": 0, "min": -1, "max": 20}),
                "x1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "y1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "z1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "id2": ("INT", {"default": 0, "min": -1, "max": 20}),
                "x2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "y2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
                "z2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": step}),
            },
            "optional":{"exp": ("EXP_DATA",),}
        }

default_edit_exp_text = """[
    [1, 0, -0.02, 0, "左眉头"],
    [2, 0, -0.02, 0, "右眉头"]
]"""

class EditExpDataByText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{                
                "text": ("STRING", {"default": default_edit_exp_text, "multiline": True}),
            },
            "optional":{"exp": ("EXP_DATA",),}
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, text: str, exp=None):
        if exp is None:
            es = ExpressionSet()
        else:
            es = ExpressionSet(es = exp)

        list_mod = json.loads(text)
        if len(list_mod) < 1:
            return (None,)

        for mod in list_mod:
            mod: list
            EditExpData.edit_idx(es, mod[0], mod[1], mod[2], mod[3])        

        return (es,)

class PrintExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "cut_noise": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.1}),
        },
            "optional": {"exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"
    OUTPUT_NODE = True

    def run(self, cut_noise, exp = None):
        if exp == None: return (exp,)

        cuted_list = []
        # luoq 尝试修复 exp-> e
        e = exp.e * 1000
        for idx in range(21):
            for r in range(3):
                a = abs(e[0, idx, r])
                if(cut_noise < a): cuted_list.append((a, e[0, idx, r], idx*10+r))

        sorted_list = sorted(cuted_list, reverse=True, key=lambda item: item[0])
        print(f"sorted_list: {[[item[2], round(float(item[1]),1)] for item in sorted_list]}")
        return (exp,)

class EditExpaData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                "expa": ("EXPA_DATA",),
                "id1": ("INT", {"default": 0, "min": -1, "max": 20}),
                "x1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 0.0001}),
                "y1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 0.0001}),
                "z1": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 0.0001}),
                "id2": ("INT", {"default": 0, "min": -1, "max": 20}),
                "x2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 0.0001}),
                "y2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 0.0001}),
                "z2": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 0.0001}),
            },
            "optional":{}
        }

    RETURN_TYPES = ("EXPA_DATA",)
    RETURN_NAMES = ("expa",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    @staticmethod
    def edit_id_pos(exp_dict, idx, x, y, z):
        if idx == -1:
            pos = exp_dict['rotation']
        else:
            pos = exp_dict['exp'][0][idx]

        pos[0] += x
        pos[1] += y
        pos[2] += z

    def run(self, expa, id1, x1, y1, z1, id2, x2, y2, z2):
        expa_new = copy.deepcopy(expa)
        
        for exp_dict in expa_new:
            self.edit_id_pos(exp_dict, id1, x1, y1, z1)
            self.edit_id_pos(exp_dict, id2, x2, y2, z2)

        return (expa_new,)

class Command:
    def __init__(self, es, change, keep):
        self.es:ExpressionSet = es
        self.change = change
        self.keep = keep

crop_factor_default = 1.7
crop_factor_min = 1.0
crop_factor_max = 2.5

class AdvancedLivePortrait:
    def __init__(self):
        self.src_images = None
        self.driving_images = None
        self.pbar = comfy.utils.ProgressBar(1)
        self.crop_factor = None

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "retargeting_eyes": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "retargeting_mouth": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "crop_factor": ("FLOAT", {"default": crop_factor_default,
                                          "min": crop_factor_min, "max": crop_factor_max, "step": 0.1}),
                "turn_on": ("BOOLEAN", {"default": True}),
                "tracking_src_vid": ("BOOLEAN", {"default": False}),
                "animate_without_vid": ("BOOLEAN", {"default": False}),
                "command": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "src_images": ("IMAGE",),
                "motion_link": ("EDITOR_LINK",),
                "driving_images": ("IMAGE",),
                "driving_action": ("EXPA_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "AdvancedLivePortrait"

    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def parsing_command(self, command, motoin_link):
        command.replace(' ', '')
        # if command == '': return
        lines = command.split('\n')

        cmd_list = []

        total_length = 0

        i = 0
        #old_es = None
        for line in lines:
            i += 1
            if line == '': continue
            try:
                cmds = line.split('=')
                idx = int(cmds[0])
                if idx == 0: es = ExpressionSet()
                else: es = ExpressionSet(es = motoin_link[idx])
                cmds = cmds[1].split(':')
                change = int(cmds[0])
                keep = int(cmds[1])
            except:
                assert False, f"(AdvancedLivePortrait) Command Err Line {i}: {line}"


                return None, None

            total_length += change + keep
            es.div(change)
            cmd_list.append(Command(es, change, keep))

        return cmd_list, total_length


    def run(self, retargeting_eyes, retargeting_mouth, turn_on, tracking_src_vid, animate_without_vid, command, crop_factor,
            src_images=None, driving_images=None, driving_action=None, motion_link=None):
        if turn_on == False: return (None,None)
        src_length = 1

        if src_images == None:
            if motion_link != None:
                self.psi_list = [motion_link[0]]
            else: return (None,None)

        if src_images != None:
            src_length = len(src_images)
            if id(src_images) != id(self.src_images) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.src_images = src_images
                if 1 < src_length:
                    self.psi_list = g_engine.prepare_source(src_images, crop_factor, True, tracking_src_vid)
                else:
                    self.psi_list = [g_engine.prepare_source(src_images, crop_factor)]


        cmd_list, cmd_length = self.parsing_command(command, motion_link)
        if cmd_list == None: return (None,None)
        cmd_idx = 0

        driving_length = 0
        if driving_images is not None:
            if id(driving_images) != id(self.driving_images):
                self.driving_images = driving_images
                self.driving_values = g_engine.prepare_driving_video(driving_images)
            driving_length = len(self.driving_values)
        elif driving_action is not None:
            self.driving_images = None
            self.driving_values = []
            for exp_dict in driving_action:
                kp_info = dict()
                e_np = np.array(exp_dict['exp'], dtype=np.float32)
                t_np = np.array(exp_dict['t'], dtype=np.float32)
                kp_info['exp'] = torch.from_numpy(e_np).float().to(get_device())
                kp_info['pitch'] = exp_dict['rotation'][0]
                kp_info['yaw'] = exp_dict['rotation'][1]
                kp_info['roll'] = exp_dict['rotation'][2]
                kp_info['scale'] = exp_dict['scale']
                kp_info['t'] = torch.from_numpy(t_np).float().to(get_device())
                self.driving_values.append(kp_info)
            driving_length = len(self.driving_values)

        total_length = max(driving_length, src_length)

        if animate_without_vid:
            total_length = max(total_length, cmd_length)

        c_i_es = ExpressionSet()
        c_o_es = ExpressionSet()
        d_0_es = None
        out_list = []

        psi = None
        pipeline = g_engine.get_pipeline()
        for i in range(total_length):

            if i < src_length:
                psi = self.psi_list[i]
                s_info = psi.x_s_info
                s_es = ExpressionSet(erst=(s_info['kp'] + s_info['exp'], torch.Tensor([0, 0, 0]), s_info['scale'], s_info['t']))

            new_es = ExpressionSet(es = s_es)

            if i < cmd_length:
                cmd = cmd_list[cmd_idx]
                if 0 < cmd.change:
                    cmd.change -= 1
                    c_i_es.add(cmd.es)
                    c_i_es.sub(c_o_es)
                elif 0 < cmd.keep:
                    cmd.keep -= 1

                new_es.add(c_i_es)

                if cmd.change == 0 and cmd.keep == 0:
                    cmd_idx += 1
                    if cmd_idx < len(cmd_list):
                        c_o_es = ExpressionSet(es = c_i_es)
                        cmd = cmd_list[cmd_idx]
                        c_o_es.div(cmd.change)
            elif 0 < cmd_length:
                new_es.add(c_i_es)

            if i < driving_length:
                d_i_info = self.driving_values[i]
                d_i_r = torch.Tensor([d_i_info['pitch'], d_i_info['yaw'], d_i_info['roll']])#.float().to(device="cuda:0")

                if d_0_es is None:
                    d_0_es = ExpressionSet(erst = (d_i_info['exp'], d_i_r, d_i_info['scale'], d_i_info['t']))

                    retargeting(s_es.e, d_0_es.e, retargeting_eyes, (11, 13, 15, 16))
                    retargeting(s_es.e, d_0_es.e, retargeting_mouth, (14, 17, 19, 20))

                new_es.e += d_i_info['exp'] - d_0_es.e
                new_es.r += d_i_r - d_0_es.r
                new_es.t += d_i_info['t'] - d_0_es.t

            r_new = get_rotation_matrix(
                s_info['pitch'] + new_es.r[0], s_info['yaw'] + new_es.r[1], s_info['roll'] + new_es.r[2])
            d_new = new_es.s * (new_es.e @ r_new) + new_es.t
            d_new = pipeline.stitching(psi.x_s_user, d_new)
            crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, d_new)
            crop_out = pipeline.parse_output(crop_out['out'])[0]

            crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb),
                                                cv2.INTER_LINEAR)
            out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(
                np.uint8)
            out_list.append(out)

            self.pbar.update_absolute(i+1, total_length, ("PNG", Image.fromarray(crop_out), None))

        if len(out_list) == 0: return (None,)

        out_imgs = torch.cat([pil2tensor(img_rgb) for img_rgb in out_list])
        return (out_imgs,)

class ExpressionEditor:
    def __init__(self):
        self.sample_image = None
        self.src_image = None
        self.crop_factor = None

    @classmethod
    def INPUT_TYPES(s):
        display = "number"
        #display = "slider"
        return {
            "required": {

                "rotate_pitch": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),
                "rotate_yaw": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),
                "rotate_roll": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),

                "blink": ("FLOAT", {"default": 0, "min": -20, "max": 5, "step": 0.5, "display": display}),
                "eyebrow": ("FLOAT", {"default": 0, "min": -10, "max": 15, "step": 0.5, "display": display}),
                "wink": ("FLOAT", {"default": 0, "min": 0, "max": 25, "step": 0.5, "display": display}),
                "pupil_x": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
                "pupil_y": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
                "aaa": ("FLOAT", {"default": 0, "min": -30, "max": 120, "step": 1, "display": display}),
                "eee": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                "woo": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                "smile": ("FLOAT", {"default": 0, "min": -0.3, "max": 1.3, "step": 0.01, "display": display}),

                "src_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": display}),
                "sample_ratio": ("FLOAT", {"default": 1, "min": -0.2, "max": 1.2, "step": 0.01, "display": display}),
                "sample_parts": (["OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"],),
                "crop_factor": ("FLOAT", {"default": crop_factor_default,
                                          "min": crop_factor_min, "max": crop_factor_max, "step": 0.1}),
            },

            "optional": {"src_image": ("IMAGE",), "motion_link": ("EDITOR_LINK",),
                         "sample_image": ("IMAGE",), "add_exp": ("EXP_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "EDITOR_LINK", "EXP_DATA")
    RETURN_NAMES = ("image", "motion_link", "save_exp")

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "AdvancedLivePortrait"

    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def run(self, rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
            src_ratio, sample_ratio, sample_parts, crop_factor, src_image=None, sample_image=None, motion_link=None, add_exp=None):
        rotate_yaw = -rotate_yaw

        new_editor_link = None
        if motion_link != None:
            self.psi = motion_link[0]
            new_editor_link = motion_link.copy()
        elif src_image != None:
            if id(src_image) != id(self.src_image) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.psi = g_engine.prepare_source(src_image, crop_factor)
                self.src_image = src_image
            new_editor_link = []
            new_editor_link.append(self.psi)
        else:
            return (None,None)

        pipeline = g_engine.get_pipeline()

        psi = self.psi
        s_info = psi.x_s_info
        #delta_new = copy.deepcopy()
        s_exp = s_info['exp'] * src_ratio
        s_exp[0, 5] = s_info['exp'][0, 5]
        s_exp += s_info['kp']

        es = ExpressionSet()

        if sample_image != None:
            if id(self.sample_image) != id(sample_image):
                self.sample_image = sample_image
                d_image_np = (sample_image * 255).byte().numpy()
                d_face = g_engine.crop_face(d_image_np[0], 1.7)
                i_d = g_engine.prepare_src_image(d_face)
                self.d_info = pipeline.get_kp_info(i_d)
                self.d_info['exp'][0, 5, 0] = 0
                self.d_info['exp'][0, 5, 1] = 0

            # "OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"
            if sample_parts == "OnlyExpression" or sample_parts == "All":
                es.e += self.d_info['exp'] * sample_ratio
            if sample_parts == "OnlyRotation" or sample_parts == "All":
                rotate_pitch += self.d_info['pitch'] * sample_ratio
                rotate_yaw += self.d_info['yaw'] * sample_ratio
                rotate_roll += self.d_info['roll'] * sample_ratio
            elif sample_parts == "OnlyMouth":
                retargeting(es.e, self.d_info['exp'], sample_ratio, (14, 17, 19, 20))
            elif sample_parts == "OnlyEyes":
                retargeting(es.e, self.d_info['exp'], sample_ratio, (1, 2, 11, 13, 15, 16))

        es.r = g_engine.calc_fe(es.e, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
                                  rotate_pitch, rotate_yaw, rotate_roll)

        if add_exp != None:
            es.add(add_exp)

        new_rotate = get_rotation_matrix(s_info['pitch'] + es.r[0], s_info['yaw'] + es.r[1],
                                         s_info['roll'] + es.r[2])
        x_d_new = (s_info['scale'] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info['t']

        x_d_new = pipeline.stitching(psi.x_s_user, x_d_new)

        crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = pipeline.parse_output(crop_out['out'])[0]

        crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)

        out_img = pil2tensor(out)

        filename = g_engine.get_temp_img_name() #"fe_edit_preview.png"
        folder_paths.get_save_image_path(filename, folder_paths.get_temp_directory())
        img = Image.fromarray(crop_out)
        img.save(os.path.join(folder_paths.get_temp_directory(), filename), compress_level=1)
        results = list()
        results.append({"filename": filename, "type": "temp"})

        new_editor_link.append(es)

        return {"ui": {"images": results}, "result": (out_img, new_editor_link, es)}

# ========== 以下为 luoq 新增 =================
class ExtractExp:

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"

    OUTPUT_NODE = True
    CATEGORY = "AdvancedLivePortrait"

    def run(self, image):

        pipeline = g_engine.get_pipeline()
                
        es = ExpressionSet()        
        d_image_np = (image * 255).byte().numpy()
        d_face = g_engine.crop_face(d_image_np[0], 1.7)
        i_d = g_engine.prepare_src_image(d_face)
        ski = pipeline.get_kp_info(i_d)
        ski['exp'][0, 5, 0] = 0
        ski['exp'][0, 5, 1] = 0

        es.e += ski['exp']
        es.r = torch.Tensor([ski['pitch'], ski['yaw'], ski['roll']])

        return (es, )

expa_data_dir = os.path.join(folder_paths.output_directory, "expa_data")
if os.path.isdir(expa_data_dir) == False:
    os.mkdir(expa_data_dir)

class ExtractExpAction:
    def __init__(self):
        self.driving_images = None
        self.driving_values = None

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "driving_images": ("IMAGE",),
                "file_name": ("STRING", {"multiline": False, "default": "expa_"}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"

    OUTPUT_NODE = True
    CATEGORY = "AdvancedLivePortrait"

    def run(self, driving_images, file_name):  

        pipeline = g_engine.get_pipeline()

        out = []
        for sample_image in driving_images:            
            drive = ExpressionSet()
            
            d_image_np = (sample_image * 255).byte().numpy()
            d_face = g_engine.crop_face(d_image_np, 1.7)
            i_d = g_engine.prepare_src_image(d_face)
            ski = pipeline.get_kp_info(i_d)
            ski['exp'][0, 5, 0] = 0
            ski['exp'][0, 5, 1] = 0

            drive.e += ski['exp']
            drive.r = torch.Tensor([ski['pitch'], ski['yaw'], ski['roll']])
            # drive.r = g_engine.calc_fe(ski['exp'], 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                             ski['pitch'], ski['yaw'], ski['roll'])
            out.append(drive.to_dict())

        with open(os.path.join(expa_data_dir, file_name + ".json"), "w") as f:
            json.dump(out, f, indent=4)

        return ()

def ajust_action_data(action_data: dict, rotate_pitch: float, rotate_yaw: float, rotate_roll: float, edge: float, mouth: float, 
                    eyes: float, other: float):
    if rotate_pitch != 1.0:        
        action_data['rotation'][0] *= rotate_pitch
    if rotate_yaw != 1.0:        
        action_data['rotation'][1] *= rotate_yaw
    if rotate_roll != 1.0:        
        action_data['rotation'][2] *= rotate_roll

    def ajust_exp_in_list(co: float, exp_list: list[int]):
        if co != 1.0:
            for i in exp_list:
                for j in range(3):
                    action_data['exp'][0][i][j] *= co
    
    ajust_exp_in_list(edge, [0, 10, 3, 4, 7])
    ajust_exp_in_list(mouth, [14, 17, 19, 20])
    ajust_exp_in_list(eyes, [1, 2, 11, 13, 15, 16])
    ajust_exp_in_list(other, [5, 6, 8, 9, 12, 18])    

    # if blink_stength !=1.0:
    #     # 上眼睑y
    #     ajust_y(11, blink_stength, True)
    #     ajust_y(15, blink_stength, True)        
    #     # 下眼睑y
    #     ajust_y(13, blink_stength, False)        
    #     ajust_y(16, blink_stength, False)        


class LoadExpActionJson:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "file_path": ("STRING", {"default": '', "multiline": False}),
            "frame_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
            "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            "rotate_pitch": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "rotate_yaw": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "rotate_roll": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "face_edge": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "mouth": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "eyes": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "other": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),            
            "every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            "force_reload": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),            
        },
        }

    RETURN_TYPES = ("EXPA_DATA", "INT")
    RETURN_NAMES = ("expa", "frame_count")
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def __init__(self) -> None:
        self.file_path = ''
        self.file_data = None

    def run(self, file_path, frame_cap, start_index, rotate_pitch, rotate_yaw, rotate_roll, face_edge, 
            mouth, eyes, other, every_nth, force_reload):
        if file_path != self.file_path or force_reload:
            if not os.path.exists(file_path):
                raise Exception('[LoadExpActionJson] wrong path file --> expa')

            with open(file_path, 'r') as f:
                self.file_data = json.load(f)
            self.file_path = file_path

        if start_index >= len(self.file_data):
            raise Exception('start_index too big!')

        # 切片
        if frame_cap == 0:
            using_data = self.file_data[start_index : : every_nth]
        else:
            using_data = self.file_data[start_index : start_index + frame_cap : every_nth]

        # exp微调
        for action_data in using_data:
            ajust_action_data(action_data, rotate_pitch, rotate_yaw, rotate_roll, face_edge, mouth, eyes, other)

        frame_count = len(using_data)
        
        return (using_data, frame_count)

class LoadSingleExpOfAction:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "file_path": ("STRING", {"default": '', "multiline": False}),
            "frame_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            "rotate_pitch": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "rotate_yaw": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "rotate_roll": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "face_edge": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "face_edge": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "mouth": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "eyes": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "other": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "force_reload": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
        },
        }

    RETURN_TYPES = ("EXP_DATA", "STRING", "INT")
    RETURN_NAMES = ("exp", "exp_string", "frame_count")
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def __init__(self) -> None:
        self.file_path = ''
        self.file_data = None

    def run(self, file_path, frame_index, rotate_pitch, rotate_yaw, rotate_roll, face_edge, mouth, 
            eyes, other, force_reload):
        if file_path != self.file_path or force_reload:
            if not os.path.exists(file_path):
                raise Exception('[LoadSingleExpOfAction] wrong path file --> expa')

            with open(file_path, 'r') as f:
                self.file_data = json.load(f)

            self.file_path = file_path

        if frame_index >= len(self.file_data):
            frame_index = len(self.file_data) - 1

        action_data = self.file_data[frame_index]

        # exp微调        
        ajust_action_data(action_data, rotate_pitch, rotate_yaw, rotate_roll, face_edge, mouth, eyes, other)

        es = ExpressionSet()
        es.from_dict(action_data)

        es_string = json.dumps(action_data, indent=4)

        frame_count = len(self.file_data)
        
        return (es, es_string, frame_count)

# default_ajust_exp_list = [
#     (11, 1, 1.3, True, '上眼睑y'),
#     (15, 1, 1.3, True, '上眼睑y'),
#     (13, 1, 1.3, False, '下眼睑y'),
#     (13, 1, 1.3, False, '下眼睑y'),
# ]
default_ajust_exp_text = """[
    [11, 1, 1.5, true, "上眼睑y"],
    [15, 1, 1.5, true, "上眼睑y"],
    [13, 1, 1.5, false, "下眼睑y"],
    [16, 1, 1.5, false, "下眼睑y"]
]"""
def adjust_exp_dict(action_data: dict, i_exp: int, i_loc: int, co: float, above_of_below_zero: bool):
    if (action_data['exp'][0][i_exp][i_loc] > 0) == above_of_below_zero:
        action_data['exp'][0][i_exp][i_loc] *= co 

class AdjustExpByText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "exp": ("EXP_DATA",),
            "text": ("STRING", {"default": default_ajust_exp_text, "multiline": True}),
        },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, exp: ExpressionSet, text):
        exp_dict = exp.to_dict()
        adjust_list = json.loads(text)
        for adjust in adjust_list:
            adjust_exp_dict(exp_dict, adjust[0], adjust[1], adjust[2], adjust[3])

        exp_new = ExpressionSet()
        exp_new.from_dict(exp_dict)
        
        return (exp_new,)

class AdjustExpaByText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "expa": ("EXPA_DATA",),
            "text": ("STRING", {"default": default_ajust_exp_text, "multiline": True}),
        },
        }

    RETURN_TYPES = ("EXPA_DATA",)
    RETURN_NAMES = ("expa",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, expa, text):
        expa_new = copy.deepcopy(expa)
        adjust_list = json.loads(text)
        for adjust in adjust_list:
            for exp_dict in expa_new:
                adjust_exp_dict(exp_dict, adjust[0], adjust[1], adjust[2], adjust[3])                
        
        return (expa_new,)

class ExpressionVideoEditor:
    def __init__(self):
        self.src_images = None
        self.driving_images = None
        self.driving_action = None
        self.pbar = comfy.utils.ProgressBar(1)        
        self.crop_factor = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_images": ("IMAGE",),                
                "src_exp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "display": "number"}),
                "drive_exp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.1, "display": "number"}),
                "retgt_brows": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "display": "number"}),
                "retgt_eyes": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "display": "number"}),
                "retgt_mouth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "display": "number"}),
                "crop_factor": ("FLOAT", {"default": crop_factor_default,
                                          "min": crop_factor_min, "max": crop_factor_max, "step": 0.1}),
                "single_mode": ("BOOLEAN", {"default": False, "label_on": "sinlge", "label_off": "muti"}),
                "single_index": ("INT", {"default": 1, "min": 0, "display": "number"}),
            },
            "optional": {                
                "driving_images": ("IMAGE",),
                "driving_action": ("EXPA_DATA",),
                "exp_neutral": ("EXP_DATA",),
                "add_exp": ("EXP_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "single_image")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "AdvancedLivePortrait"

    def run(self, src_images, src_exp, drive_exp, retgt_brows, retgt_eyes, retgt_mouth, 
            crop_factor, single_mode, single_index, driving_images=None, driving_action=None, 
            exp_neutral: ExpressionSet=None, add_exp: ExpressionSet=None):
        
        src_length = len(src_images)
        if id(src_images) != id(self.src_images) or self.crop_factor != crop_factor:
            self.crop_factor = crop_factor
            self.src_images = src_images
            if 1 < src_length:
                self.psi_list = g_engine.prepare_source(src_images, crop_factor, True)
            else:
                self.psi_list = [g_engine.prepare_source(src_images, crop_factor)]

        driving_length = 0
        if driving_images is not None:
            if id(driving_images) != id(self.driving_images):
                self.driving_images = driving_images
                self.driving_action = None
                self.driving_values = g_engine.prepare_driving_video(driving_images)
            driving_length = len(self.driving_values)
        elif driving_action is not None:
            if id(driving_action) != id(self.driving_action):
                self.driving_images = None
                self.driving_action = driving_action
                self.driving_values = []
                for exp_dict in driving_action:
                    kp_info = dict()
                    e_np = np.array(exp_dict['exp'], dtype=np.float32)
                    t_np = np.array(exp_dict['t'], dtype=np.float32)
                    kp_info['exp'] = torch.from_numpy(e_np).float().to(get_device())
                    kp_info['pitch'] = exp_dict['rotation'][0]
                    kp_info['yaw'] = exp_dict['rotation'][1]
                    kp_info['roll'] = exp_dict['rotation'][2]
                    kp_info['scale'] = exp_dict['scale']
                    kp_info['t'] = torch.from_numpy(t_np).float().to(get_device())
                    self.driving_values.append(kp_info)
            driving_length = len(self.driving_values)

        if src_length > driving_length:
            raise Exception('src_images count < driving count, excute failed!')

        final_length = driving_length

        d_0_es = None
        out_list = []

        psi = None
        pipeline = g_engine.get_pipeline()

        # 不同模式，使用不同的 i 迭代器
        if single_mode:
            iterator = iter([0, single_index])
        else:
            iterator = range(final_length)

        for i in iterator:
            
            i_src = i % src_length  # src_image 循环使用
            psi = self.psi_list[i_src]
            s_info = psi.x_s_info
            s_es = ExpressionSet(erst=(s_info['kp'] + s_info['exp'] * src_exp, 
                                torch.Tensor([0, 0, 0]), s_info['scale'], s_info['t']))

            new_es = ExpressionSet(es = s_es)    
            
            d_i_info = self.driving_values[i]
            d_i_r = torch.Tensor([d_i_info['pitch'], d_i_info['yaw'], d_i_info['roll']])#.float().to(device="cuda:0")

            if d_0_es is None:                
                # if exp_neutral:
                    # 表情部分，以neutral为准
                #     d_0_es = ExpressionSet(erst = (exp_neutral.e, d_i_r, d_i_info['scale'], d_i_info['t']))
                # else:
                #     d_0_es = ExpressionSet(erst = (d_i_info['exp'], d_i_r, d_i_info['scale'], d_i_info['t']))
                d_0_es = ExpressionSet(es=exp_neutral)

            # 重定向，即：某组骨骼，使用 drive第一帧的参数
            if retgt_brows > 0:
                retargeting(new_es.e, d_0_es.e, retgt_brows, (1, 2))
            if retgt_eyes > 0:
                retargeting(new_es.e, d_0_es.e, retgt_eyes, (11, 13, 15, 16))
            if retgt_mouth > 0:
                retargeting(new_es.e, d_0_es.e, retgt_mouth, (14, 17, 19, 20))

            new_es.e += d_i_info['exp'] * drive_exp - d_0_es.e * drive_exp
            new_es.r += d_i_r - d_0_es.r
            new_es.t += d_i_info['t'] - d_0_es.t

            if add_exp:
                new_es.add(add_exp)

            r_new = get_rotation_matrix(
                s_info['pitch'] + new_es.r[0],
                s_info['yaw'] + new_es.r[1], 
                s_info['roll'] + new_es.r[2]
            )
            d_new = new_es.s * (new_es.e @ r_new) + new_es.t
            d_new = pipeline.stitching(psi.x_s_user, d_new)
            crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, d_new)
            crop_out = pipeline.parse_output(crop_out['out'])[0]

            crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb),
                                                cv2.INTER_LINEAR)
            out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(
                np.uint8)
            out_list.append(out)

            self.pbar.update_absolute(i+1, final_length, ("PNG", Image.fromarray(crop_out), None))

        if len(out_list) == 0: return (None,)

        out_imgs = torch.cat([pil2tensor(img_rgb) for img_rgb in out_list])
        last_img = pil2tensor(out_list[-1])

        return (out_imgs, last_img)

class LoadBLBRequestInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {          
            "name_project": ("STRING", {"default": '', "multiline": False}),
            "clip_index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
            "dir_src_root": ("STRING", {"default": 'D:\\Blender workspace\\BLB', "multiline": False}),
            "dir_project_root": ("STRING", {"default": 'F:\\My Work\\Breathing LookBook\\projects\\', "multiline": False}),
            "fps": ("INT", {"default": 60, "min": 1, "max": 120, "step": 1}),
            "force_reload": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
        },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("name_clip", "video_src", "drive_cap", "drive_start", "seek_seconds")
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def __init__(self) -> None:
        self.file_path = ''
        self.file_data = None
        self.clip_index = -1
        self.clip_info = None

    def run(self, name_project: str, clip_index: int, dir_src_root: str, dir_project_root: str, fps: int, force_reload: bool):
        file_reloaded = False

        # 加载配置文件
        path_request_info = os.path.join(dir_project_root,name_project, 'request_clips_info.json')
        if path_request_info != self.file_path or force_reload:
            if not os.path.exists(path_request_info):
                raise Exception('[LoadBLBRequestInfoFile] wrong path file --> request_info.json')

            with open(path_request_info, 'r') as f:
                self.file_data = json.load(f)
                file_reloaded = True

            self.file_path = path_request_info

        # 定位clip
        if clip_index != self.clip_index or file_reloaded:
            name_clip = 'clip_{:03}'.format(clip_index)
            self.clip_info = None
            
            for info in self.file_data:
                info: dict
                if info['name'] == name_clip:
                    self.clip_info = info.copy()
                    break

            if self.clip_info is None:
                raise Exception('[LoadBLBRequestInfoFile]读取clip失败。path={}, clip_index={}'.format(
                    self.file_path,
                    clip_index
                ))
            
            self.clip_index = clip_index

        # src视频信息
        name_vidoe = '{:03}.mp4'.format(clip_index)
        path_video_src = os.path.join(dir_src_root, name_project, 'blender', name_clip, 'output', name_vidoe)

        # drive信息
        driver_cap = self.clip_info['length']
        driver_start = self.clip_info['frame_start'] - 1
        seek_seconds = 1.0 / fps * driver_start

        return (name_clip, path_video_src, driver_cap, driver_start, seek_seconds)


NODE_CLASS_MAPPINGS = {
    "AdvancedLivePortrait": AdvancedLivePortrait,
    "ExpressionEditor": ExpressionEditor,
    "LoadExpData": LoadExpData,
    "LoadExpDataJson": LoadExpDataJson,
    "LoadExpDataString": LoadExpDataString,
    "SaveExpData": SaveExpData,
    "ExpData": ExpData,
    "ShowExpData": ShowExpData,
    "EditExpData": EditExpData,
    "EditExpDataRough": EditExpDataRough,
    "EditExpDataByText": EditExpDataByText,
    # "EditExpaData": EditExpaData,   # 改了半天没用，融不进动图
    "PrintExpData:": PrintExpData,
    "ExtractExpAction:": ExtractExpAction,
    "ExtractExp:": ExtractExp,
    "LoadExpActionJson:": LoadExpActionJson,
    "ExpressionVideoEditor:": ExpressionVideoEditor,
    "LoadSingleExpOfAction:": LoadSingleExpOfAction,
    "LoadBLBRequestInfo": LoadBLBRequestInfo,
    "AdjustExpByText": AdjustExpByText,
    "AdjustExpaByText": AdjustExpaByText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLivePortrait": "Advanced Live Portrait (PHM)",
    "ExpressionEditor": "Expression Editor (PHM)",
    "LoadExpData": "Load Exp Data (PHM)",
    "LoadExpDataJson": "Load Exp Data Json",
    "LoadExpDataString": "Load Exp Data String",
    "SaveExpData": "Save Exp Data (PHM)",
    "ExtractExpAction": "Extract Exp Action",
    "LoadExpActionJson": "Load Exp Action",
    "ExpressionVideoEditor": "Expression Video Editor",
    "LoadSingleExpOfAction": "Load Single Exp Of Action",
    "LoadBLBRequestInfo": "Load BLB Request Info",
    "AdjustExpByText": "Adjust Exp By Text",
    "AdjustExpaByText": "Adjust Expa By Text",
}
