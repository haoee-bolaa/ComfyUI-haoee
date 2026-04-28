import os
import torch
import requests
import time
from PIL import Image
from io import BytesIO
import json
import comfy.utils
import re
import base64
import uuid
import folder_paths
import cv2
import shutil
import subprocess
import traceback
from .utils import pil2tensor, tensor2pil
from comfy.comfy_types import IO

baseurl = "https://maas.haoee.com"


def _image_tensor_to_base64(image_tensor, with_prefix=True):
    if image_tensor is None:
        return None
    pil_image = tensor2pil(image_tensor)[0]
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if with_prefix:
        return f"data:image/png;base64,{b64}"
    return b64


class ComflyVideoAdapter:
    def __init__(self, video_path_or_url):
        if video_path_or_url.startswith('http'):
            self.is_url = True
            self.video_url = video_path_or_url
            self.video_path = None
        else:
            self.is_url = False
            self.video_path = video_path_or_url
            self.video_url = None
        
    def get_dimensions(self):
        if self.is_url:
            return 1280, 720
        else:
            try: 
                cap = cv2.VideoCapture(self.video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return width, height
            except Exception as e:
                print(f"Error getting video dimensions: {str(e)}")
                return 1280, 720
            
    def _remux_faststart(self, input_path, output_path):
        """Use ffmpeg to remux with moov atom at the front for seekability, duration and thumbnail."""
        try:
            if hasattr(folder_paths, "get_ffmpeg_path"):
                ffmpeg_path = folder_paths.get_ffmpeg_path()
            else:
                ffmpeg_path = shutil.which("ffmpeg")

            if not ffmpeg_path:
                print("[ComflyVideoAdapter] ffmpeg not found, skipping faststart remux")
                return False

            result = subprocess.run(
                [ffmpeg_path, "-y", "-i", input_path, "-c", "copy", "-movflags", "+faststart", output_path],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                print(f"[ComflyVideoAdapter] ffmpeg remux failed: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print("[ComflyVideoAdapter] ffmpeg remux timed out")
            return False
        except Exception as e:
            print(f"[ComflyVideoAdapter] ffmpeg remux error: {str(e)}")
            return False

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if self.is_url:
            try:
                response = requests.get(self.video_url, stream=True)
                response.raise_for_status()

                temp_path = output_path + ".tmp"
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if self._remux_faststart(temp_path, output_path):
                    os.remove(temp_path)
                else:
                    shutil.move(temp_path, output_path)
                return True
            except Exception as e:
                print(f"Error downloading video from URL: {str(e)}")
                if os.path.exists(output_path + ".tmp"):
                    os.remove(output_path + ".tmp")
                return False
        else:
            try:
                shutil.copyfile(self.video_path, output_path)
                return True
            except Exception as e:
                print(f"Error saving video: {str(e)}")
                return False


class Comfly_Haoee_api_key:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apikey": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("apikey",)
    FUNCTION = "set_api_base"
    CATEGORY = "好易"

    def set_api_base(self, apikey=""):
        return (apikey,)


class Comfly_HaoeeVideo_MiniMax:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["MiniMax-Hailuo-2.3-Fast", "MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02"], {"default": "MiniMax-Hailuo-02"}),
                "duration": (["6", "10"], {"default": "6"}),
                "resolution": (["768P", "1080P"], {"default": "768P"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 300
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, prompt, model="MiniMax-Hailuo-02", duration="6", resolution="768P", prompt_optimizer=True, image=None, api_key="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "错误，未配置api_key"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))

        if image is None:
            error_message = "错误，未配置image"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            payload = {
                "model": model,
                "prompt": prompt,
                "duration": int(duration),
                "resolution": resolution,
                "first_frame_image": image_base64,
                "prompt_optimizer": prompt_optimizer,
                "seed": seed if seed > 0 else 0
            }
            
            response = requests.post(
                f"{baseurl}/api/v2/hailuo/v1/video_generation", 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            task_id = result.get("task_id")

            if not task_id:
                error_message = "错误，未获取到task_id"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(30)
            print(f"Video generation task submitted. Task ID: {task_id}")

            max_attempts = 60  
            attempts = 0
            file_id = None
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)  
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/api/v2/get_task/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")
                    
                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_result = status_response.json()
                    state = status_result["data"]["state"]
                    
                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        data = status_result.get("data", {})
                        data_info = data.get("data_info", {}).get("data", {})
                        video_url = None
                        # 优先 more_file_info
                        more_file = data_info.get("more_file_info")
                        if more_file and "download_url" in more_file:
                            video_url = more_file["download_url"]
                            file_id = more_file["file_id"]
                        # 兜底 file_info[0]
                        if not video_url:
                            file_list = data_info.get("file_info", [])
                            if file_list and "file_url" in file_list[0]:
                                video_url = file_list[0]["file_url"]

                        if not video_url:
                            return (
                                None,
                                task_id,
                                json.dumps(status_result, ensure_ascii=False)
                            )
                        break
                    elif state == "failed":
                        error_message = f"Video generation failed: {status_result.get('base_resp', {}).get('status_msg', 'Unknown error')}"
                        print(error_message)
                        raise Exception(error_message)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error checking generation status: {str(e)}")
            pbar.update_absolute(100)
            if not video_url:
                return (
                    None,
                    task_id,
                    json.dumps(status_result, ensure_ascii=False)
                )
            print(f"Video generation completed. URL: {video_url}")
            
            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "file_id": file_id,
                "video_url": video_url,
            }
            
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            traceback.print_exc()
            raise Exception(error_message)


class Comfly_HaoeeVideo_Sora2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["sora-2", "sora-2-pro"], {"default": "sora-2"}),
                "seconds": (["4", "8", "12"],{"default": "4"}),
                "size": (["720x1280","1280x720","1024x1792","1792x1024"], {"default":"720x1280"}),
                "apikey": ("STRING", {"default": ""})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "process"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 300
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def get_image_size(self, image):
        """
        image: ComfyUI IMAGE tensor
        return: (width, height)
        """
        if image is None:
            return None

        _, height, width, _ = image.shape
        return (width, height)

    def process(self, prompt, model,  seconds="4", size="720x1280", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))
            
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            raise Exception(error_message)

        width, height = self.get_image_size(image)

        if (width, height) not in [(1280, 720), (720, 1280), (1024, 1792), (1792, 1024)]:
            error_message = "图片尺寸必须为 1280x720, 720x1280, 1024x1792, or 1792x1024"
            print(error_message)
            raise Exception(error_message)

        if model == "sora-2" and size not in ["720x1280", "1280x720"]:
            error_message = "sora-2模型只支持720x1280和1280x720尺寸"
            print(error_message)
            raise Exception(error_message)
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            
            form_data = {
                "prompt": prompt,
                "model": model,
                "seconds": str(seconds),
                "size": str(size),
                # "seed": seed if seed > 0 else 0
            }
            files = {
                "input_reference": ("image.png", base64.b64decode(image_base64.split(",")[1]), "image/png")
            }
            response = requests.post(
                f"{baseurl}/v1/videos",
                headers=headers,
                data=form_data,
                files=files,
                timeout=self.timeout
            )
            pbar.update_absolute(20)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            task_id = result.get("id")
            
            if not task_id:
                error_message = "No task ID in API response"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(30)
            print(f"Video generation task submitted. Task ID: {task_id}")

            max_attempts = 80  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(5)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/v1/videos/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")
                    
                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_data = status_response.json()
                    status = status_data.get("status")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    #queued、success、in_progress、failed、completed
                    if status == "completed":
                        content_response = requests.get(
                            f"{baseurl}/v1/videos/{task_id}/content",
                            headers=headers,
                            stream=True,
                            timeout=self.timeout
                        )
                        content_type = content_response.headers.get("Content-Type", "")
                        # 如果是视频流
                        if "video" in content_type or "octet-stream" in content_type:
                            output_dir = folder_paths.get_output_directory()
                            filename = f"sora_{uuid.uuid4().hex}.mp4"
                            file_path = os.path.join(output_dir, filename)
                            with open(file_path, "wb") as f:
                                for chunk in content_response.iter_content(8192):
                                    if chunk:
                                        f.write(chunk)
                            print("Video saved:", file_path)
                            video_url = file_path
                            break
                        # 如果是 JSON
                        else:
                            try:
                                content_data = content_response.json()
                                video_url = content_data.get("url", "")
                            except:
                                video_url = ""

                            if video_url:
                                print("Video URL ready:", video_url)
                                break
                            else:
                                print("Content not ready, waiting 3s...")
                                time.sleep(3)
                    elif status == "failed":
                        fail_reason = status_data.get("error", {}).get("message", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        raise Exception(error_message)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error checking task status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to get video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)
            
            video_adapter = ComflyVideoAdapter(video_url)
            
            pbar.update_absolute(100)
            
            response_data = {
                "status": "success",
                "model": model,
                "prompt": prompt,
                "seconds": seconds,
                "size": size,
                "task_id": task_id,
                "video_url": video_url
            }
            
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))
            
        except Exception as e:
            error_message = f"Error in video generation: {str(e)}"
            print(error_message)
            traceback.print_exc()
            raise Exception(error_message)


class Comfly_HaoeeVideo_Kling:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["kling-video-o1", "kling-v2-6", "kling-video-v2-5-turbo", "kling-v2-1-master"], {"default": "kling-v2-6"}),
                "duration": (["5", "10"], {"default": "5"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_tail": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "mode": (["std", "pro"], {"default": "std"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aspect_ratio": (["16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "16:9"}),
                "sound": (["on", "off"], {"default": "off"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 300

    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=False)

    def generate_video(self, image, prompt, model, duration, api_key, negative_prompt="", seed=0, image_tail=None, **kwargs):
        if api_key.strip():
            self.api_key = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))

        if image is None:
            error_message = "Image not provided"
            print(error_message)
            raise Exception(error_message)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            model_map = {
                "kling-video-o1": "kling-video-o1",
                "kling-v2-6": "kling-v2-6",
                "kling-video-v2-5-turbo": "kling-v2-5-turbo",
                "kling-v2-1-master": "kling-v2-1-master"
            }

            mode = kwargs.get("mode", "std")
            sound = kwargs.get("sound", "off")
            cfg_scale = kwargs.get("cfg_scale", 0.5)
            aspect_ratio = kwargs.get("aspect_ratio", "auto")

            payload = {
                "model_name": model_map.get(model, model),
                "prompt": prompt,
                "duration": duration,
            }

            if negative_prompt:
                payload["negative_prompt"] = negative_prompt

            if model == "kling-video-o1":
                # o1: image_list, mode, sound, aspect_ratio
                payload["mode"] = mode
                payload["sound"] = sound
                payload["aspect_ratio"] = aspect_ratio
                payload["image_list"] = [{"image_url": image_base64, "type": "first_frame"}]
                if image_tail is not None:
                    payload["image_list"].append({"image_url": self.image_to_base64(image_tail), "type": "end_frame"})

            elif model == "kling-v2-6":
                # v2-6: image, image_tail, mode, sound, cfg_scale
                payload["image"] = image_base64
                payload["mode"] = mode if mode != "std" else "pro"
                payload["sound"] = sound
                payload["cfg_scale"] = cfg_scale
                if image_tail is not None:
                    payload["image_tail"] = self.image_to_base64(image_tail)

            elif model == "kling-video-v2-5-turbo":
                # v2-5-turbo: image, image_tail, mode, cfg_scale
                payload["image"] = image_base64
                payload["mode"] = mode
                payload["cfg_scale"] = cfg_scale
                if image_tail is not None:
                    payload["image_tail"] = self.image_to_base64(image_tail)

            elif model == "kling-v2-1-master":
                # v2-1-master: image, image_tail, cfg_scale
                payload["image"] = image_base64
                payload["cfg_scale"] = cfg_scale
                if image_tail is not None:
                    payload["image_tail"] = self.image_to_base64(image_tail)

            if seed > 0:
                payload["seed"] = seed

            if model == "kling-video-o1":
                response = requests.post(
                    f"{baseurl}/kling/v1/videos/omni-video",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
            else:
                response = requests.post(
                    f"{baseurl}/kling/v1/videos/image2video",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
            pbar.update_absolute(20)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")

            if response.status_code != 200:
                error_message = f"Error: {response.status_code} {response.reason} - {response.text}"
                error_response = {"task_status": "failed", "task_status_msg": error_message}
                raise Exception(json.dumps(error_response, ensure_ascii=False))
            
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                raise Exception(json.dumps(error_response, ensure_ascii=False))
                
            task_id = result["data"]["task_id"]
            
            if not task_id:
                error_message = "No task ID in API response"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(30)
            print(f"Video generation task submitted. Task ID: {task_id}")
            
            max_attempts = 60  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1

                try:
                    if model == "kling-video-o1":
                        status_response = requests.get(
                            f"{baseurl}/kling/v1/images/omni-image/{task_id}",
                            headers=headers,
                            timeout=self.timeout
                        )
                    else:     
                        status_response = requests.get(
                            f"{baseurl}/kling/v1/videos/image2video/{task_id}",
                            headers=headers,
                            timeout=self.timeout
                        )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")

                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_data = status_response.json()
                    status = status_data["data"]["task_status"]

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if status == "succeed":
                        video_url = status_data["data"]["task_result"]["videos"][0]["url"]
                        break
                            
                    elif status == "failed":
                        fail_reason = status_data["data"].get("task_status_msg", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        raise Exception(error_message)

                except requests.exceptions.RequestException as e:
                    print(f"Error checking task status: {str(e)}")

            if not video_url:
                error_message = f"Failed to get video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)
            
            video_adapter = ComflyVideoAdapter(video_url)
            
            pbar.update_absolute(100)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model_name": model,
                "duration": duration,
                "mode": mode,
                "video_url": video_url
            }
            
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))

        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error generating video: {str(e)}"}
            print(f"Error generating video: {str(e)}")
            raise Exception(json.dumps(error_response, ensure_ascii=False))


class Comfly_HaoeeVideo_vidu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["viduq2-pro", "viduq2-turbo", "viduq2"], {"default": "viduq2-pro"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ([5, 10], {"default": 5}),
                "resolution": (["540p", "720p", "1080p"], {"default": "720p"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "is_rec": ("BOOLEAN", {"default": False}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 300
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, image, model="viduq2-pro", prompt="", api_key="", is_rec=False, duration=5, seed=0, resolution="720p", 
                      movement_amplitude="auto", bgm=False):
        
        if api_key.strip():
            self.api_key = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))
        
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],  
                "duration": duration,
                "resolution": resolution,
                "is_rec": is_rec,
                "bgm": bgm,
                "movement_amplitude": movement_amplitude,
            }
            if seed > 0:
                payload["seed"] = seed

            response = requests.post(
                f"{baseurl}/ent/v2/img2video",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            task_id = result.get("task_id")
            
            if not task_id:
                error_message = "No task ID in API response"
                print(error_message)
                raise Exception(error_message)
                
            pbar.update_absolute(30)
            print(f"Video generation task submitted. Task ID: {task_id}")

            max_attempts = 60
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/ent/v2/tasks/{task_id}/creations",
                        headers=headers,
                        timeout=self.timeout
                    )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")
                    
                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_result = status_response.json()
                    state = status_result.get("state", "")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        raise Exception(error_message)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(100)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "seed": result.get("seed", seed)
            }
            
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            traceback.print_exc()
            raise Exception(error_message)


class Comfly_HaoeeVideo_Veo3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["veo3.1-fast", "veo3.1", "veo3"], {"default": "veo3"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "apikey": ("STRING", {"default": ""})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 300
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))
    
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            payload = {
                "prompt": prompt,
                "model": model,
                "enhance_prompt": enhance_prompt,
                "aspect_ratio": aspect_ratio,
                "images": [image_base64],
                "seed": seed if seed > 0 else 0
            }

            response = requests.post(
                f"{baseurl}/v2/videos/generations",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            task_id = result.get("task_id")
                
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(30)

            max_attempts = 60  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/v2/videos/generations/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")
                    
                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_result = status_response.json()
                    status = status_result.get("status", "")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        if "data" in status_result and "output" in status_result["data"]:
                            video_url = status_result["data"]["output"]
                            break
                    elif status == "FAILURE":
                        fail_reason = status_result.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        raise Exception(error_message)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(100)
            print(f"Video generation completed. URL: {video_url}")
            
            response_data = {
                "code": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "enhance_prompt": enhance_prompt,
                "aspect_ratio": aspect_ratio,
                "video_url": video_url,
            }
            
            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            raise Exception(error_message)
        

class Comfly_HaoeeVideo_Wan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["wan2.6-i2v-flash", "wan2.6-i2v"], {"default": "wan2.6-i2v-flash"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "resolution": (["720P", "1080P"], {"default": "720P"}),
                "duration": (["5", "10", "15"], {"default": "5"}),
                "prompt_extend": ("BOOLEAN", {"default": False}),
                "shot_type": (["single", "multi"], {"default": "single"}),
                "audio": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 300
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, model, prompt, negative_prompt, resolution="720P", duration="5", prompt_extend=False, shot_type="single", audio=False, watermark=False, apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))
    
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            raise Exception(error_message)

        if model == "wan2.6-i2v" and not audio:
            audio = True
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            payload = {
                "model": model,
                "input": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt if negative_prompt else "",
                    "img_url": image_base64
                },
                "parameters": {
                    "resolution": resolution,
                    "duration": duration,
                    "prompt_extend": prompt_extend,
                    "shot_type": shot_type,
                    "audio": audio,
                    "watermark": watermark,
                    "seed": seed if seed > 0 else 0
                }
            }

            response = requests.post(
                f"{baseurl}/api/v1/services/aigc/video-generation/video-synthesis",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            task_id = result.get("output", {}).get("task_id")
                
            if not task_id:
                error_message = result.get("message", "No task ID returned from API") 
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(30)

            max_attempts = 60  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/api/v1/tasks/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")
                    
                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_result = status_response.json()
                    status = status_result.get("output", {}).get("task_status")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status == "SUCCEEDED":
                        video_url = status_result.get("output", {}).get("video_url")
                        break
                    elif status == "FAILED":
                        fail_reason = status_result.get("output", {}).get("message", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        raise Exception(error_message)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(100)
            print(f"Video generation completed. URL: {video_url}")
            
            response_data = {
                "code": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "duration": duration,
                "negative_prompt": negative_prompt,
                "prompt_extend": prompt_extend,
                "video_url": video_url,
            }
            
            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            raise Exception(error_message)
   
        
def safe_video_adapter(video_url=None):
    if not video_url:
        return None
    try:
        return ComflyVideoAdapter(video_url)
    except Exception as e:
        print(f"[VideoAdapter] fallback to empty video: {e}")
        return None


class Comfly_HaoeeVideo_Doubao:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": ([
                    "doubao-seedance-1-0-pro-250528",
                    "doubao-seedance-1-0-lite-i2v-250428",
                    "doubao-seedance-1-5-pro-251215",
                    "doubao-seedance-1-0-pro-fast-251015"
                ], {"default": "doubao-seedance-1-0-pro-250528"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": ([5, 10], {"default": 5}),
                "ratio": (["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21", "keep_ratio", "adaptive"], {"default": "16:9"}),
                "apikey": ("STRING", {"default": ""})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 30  # GET 轮询超时，避免300秒阻塞
        self.api_key = None

    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)

    def generate_video(self, prompt, model, resolution="720p", duration=5, ratio="16:9", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))

        if image is None:
            error_message = "Image not provided"
            print(error_message)
            raise Exception(error_message)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            payload = {
                "model": model,
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ],
                "resolution": resolution,
                "duration": int(duration),
                "ratio": ratio
            }

            if seed > 0:
                payload["seed"] = seed

            response = requests.post(
                f"{baseurl}/volc/v1/contents/generations/tasks",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            # print(f"Request sent to {response.url}. Status code: {response.status_code}, Response: {response.text}")

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)

            result = response.json()
            print(f"result: {result}")

            task_id = result.get("id")
            video_url = result.get("content", {}).get("video_url")  # POST 同步返回

            # 如果同步返回 video_url，直接返回
            if video_url:
                pbar.update_absolute(100)
                print(f"Video generated (sync). URL: {video_url}")
                video_adapter = safe_video_adapter(video_url)
                return (video_adapter, task_id, json.dumps(result, ensure_ascii=False))

            # 如果 video_url 没有返回，则进入轮询（异步模型）
            pbar.update_absolute(30)
            max_attempts = 60
            attempts = 0

            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/volc/v1/contents/generations/tasks/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    print(f"Status check ({attempts}): {status_response.status_code}, Response: {status_response.text}")

                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                    status_result = status_response.json()
                    print(f"Response: {status_result}")
                    status = status_result.get("status", "").lower()
                    video_url = status_result.get("content", {}).get("video_url")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status in ["succeeded", "success"] and video_url:
                        print(f"Video generated (async). URL: {video_url}")
                        break
                    elif status in ["failed", "failure"]:
                        fail_reason = status_result.get("fail_reason", "Unknown error")
                        raise Exception(f"Video generation failed: {fail_reason}")

                except requests.exceptions.RequestException as e:
                    print(f"Error checking generation status: {str(e)}")

            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)

            pbar.update_absolute(100)
            video_adapter = safe_video_adapter(video_url)
            response_data = {
                "code": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "duration": int(duration),
                "ratio": ratio,
                "video_url": video_url
            }
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))

        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            raise Exception(error_message)


class Comfly_HaoeeVideo_haoeedance:
    HAOEEDANCE_CREATE_URL = f"{baseurl}/api/v3/contents/generations/tasks"
    HAOEEDANCE_QUERY_URL = f"{baseurl}/api/v3/contents/generations/tasks/{{id}}"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["haoeedance-2-0", "haoeedance-2-0-fast"], {"default": "haoeedance-2-0"}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1}),
                "ratio": (["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"], {"default": "adaptive"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_video_url": ("STRING", {"default": ""}),
                "reference_audio_url": ("STRING", {"default": ""}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("video", "last_frame", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 30
        self.api_key = None

    def _img_b64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)

    def _empty_image(self):
        return torch.zeros(1, 1, 1, 3)

    def _download_last_frame(self, url):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            pil_img = Image.open(BytesIO(resp.content))
            return pil2tensor(pil_img)
        except Exception as e:
            print(f"[haoeedance] download last_frame failed: {e}")
            return self._empty_image()

    def generate_video(
        self,
        prompt,
        model,
        resolution="720p",
        duration=5,
        ratio="adaptive",
        apikey="",
        first_frame=None,
        last_frame=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_video_url="",
        reference_audio_url="",
        generate_audio=True,
        watermark=False,
        return_last_frame=False,
    ):
        if apikey and apikey.strip():
            self.api_key = apikey.strip()

        if not self.api_key:
            error_response = {"code": "error", "message": "API key not provided"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))

        prompt_preview = (prompt[:80] + "...") if prompt and len(prompt) > 80 else (prompt or "")
        print(
            f"[haoeedance] call: model={model}, resolution={resolution}, duration={duration}, "
            f"ratio={ratio}, generate_audio={generate_audio}, watermark={watermark}, "
            f"return_last_frame={return_last_frame}, "
            f"first_frame={first_frame is not None}, last_frame={last_frame is not None}, "
            f"ref_imgs={sum(x is not None for x in (reference_image_1, reference_image_2, reference_image_3, reference_image_4))}, "
            f"ref_video={'yes' if reference_video_url and reference_video_url.strip() else 'no'}, "
            f"ref_audio={'yes' if reference_audio_url and reference_audio_url.strip() else 'no'}, "
            f"prompt={prompt_preview!r}"
        )

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelname": model,
            }

            content = []
            if prompt and prompt.strip():
                content.append({"type": "text", "text": prompt})

            if first_frame is not None:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._img_b64(first_frame)},
                    "role": "first_frame",
                })
            if last_frame is not None:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._img_b64(last_frame)},
                    "role": "last_frame",
                })

            for ref_img in (reference_image_1, reference_image_2, reference_image_3, reference_image_4):
                if ref_img is not None:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": self._img_b64(ref_img)},
                        "role": "reference_image",
                    })

            if reference_video_url and reference_video_url.strip():
                content.append({
                    "type": "video_url",
                    "video_url": {"url": reference_video_url.strip()},
                    "role": "reference_video",
                })

            if reference_audio_url and reference_audio_url.strip():
                content.append({
                    "type": "audio_url",
                    "audio_url": {"url": reference_audio_url.strip()},
                    "role": "reference_audio",
                })

            if not content:
                raise Exception("content 为空：至少需要 prompt 或一张图片/视频/音频")

            content_summary = [
                {"type": item["type"], "role": item.get("role", "")} for item in content
            ]
            print(f"[haoeedance] content items ({len(content)}): {content_summary}")

            payload = {
                "model": model,
                "content": content,
                "resolution": resolution,
                "duration": int(duration),
                "ratio": ratio,
                "generate_audio": bool(generate_audio),
                "watermark": bool(watermark),
                "return_last_frame": bool(return_last_frame),
            }

            print(f"[haoeedance] POST {self.HAOEEDANCE_CREATE_URL}")
            response = requests.post(
                self.HAOEEDANCE_CREATE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            pbar.update_absolute(20)
            print(f"[haoeedance] create status={response.status_code}, body={response.text}")

            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")

            result = response.json()
            task_id = result.get("id")
            if not task_id:
                raise Exception(f"未获取到 task id, response={result}")

            pbar.update_absolute(30)
            print(f"[haoeedance] task submitted: {task_id}")

            max_attempts = 60
            attempts = 0
            video_url = None
            last_frame_url = None
            status_result = {}

            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        self.HAOEEDANCE_QUERY_URL.format(id=task_id),
                        headers=headers,
                        timeout=self.timeout,
                    )
                    print(f"[haoeedance] poll {attempts}: status={status_response.status_code}, body={status_response.text}")

                    if status_response.status_code != 200:
                        raise Exception(f"Status check failed: {status_response.status_code} - {status_response.text}")

                    status_result = status_response.json()
                    task_status = (status_result.get("status") or "").lower()
                    content_resp = status_result.get("content") or {}

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    print(f"[haoeedance] task {task_id} status={task_status} (attempt {attempts}/{max_attempts})")

                    if task_status == "succeeded":
                        video_url = content_resp.get("video_url")
                        last_frame_url = content_resp.get("last_frame_url")
                        if video_url:
                            print(f"[haoeedance] task {task_id} succeeded, last_frame_url={'yes' if last_frame_url else 'no'}")
                            break
                        raise Exception(f"任务成功但未返回 video_url, response={status_result}")
                    elif task_status in ("failed", "expired", "cancelled"):
                        err = status_result.get("error") or {}
                        err_msg = err.get("message") if isinstance(err, dict) else str(err)
                        print(f"[haoeedance] task {task_id} {task_status}: {err_msg or 'Unknown error'}")
                        raise Exception(f"Video generation {task_status}: {err_msg or 'Unknown error'}")

                except requests.exceptions.RequestException as e:
                    print(f"[haoeedance] poll request error: {e}")

            if not video_url:
                raise Exception(f"轮询 {max_attempts} 次后仍未获取到 video_url")

            pbar.update_absolute(90)

            if return_last_frame and last_frame_url:
                print(f"[haoeedance] downloading last_frame: {last_frame_url}")
                last_frame_tensor = self._download_last_frame(last_frame_url)
            else:
                last_frame_tensor = self._empty_image()

            pbar.update_absolute(100)
            print(f"[haoeedance] done. task_id={task_id}, video_url={video_url}")

            video_adapter = safe_video_adapter(video_url)
            return (video_adapter, last_frame_tensor, task_id, json.dumps(status_result, ensure_ascii=False))

        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            traceback.print_exc()
            raise Exception(error_message)


class Comfly_HaoeeVideo_grok:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["grok-video-3"], {"default": "grok-video-3"}),
                "aspect_ratio": (["2:3", "3:2", "1:1"], {"default": "2:3"}),
                "size": (["720P"], {"default": "720P"}),
                "apikey": ("STRING", {"default": ""})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = 300
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, prompt, model="grok-video-3", aspect_ratio="2:3", size="720P", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            raise Exception(json.dumps(error_response, ensure_ascii=False))
    
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            image_base64 = self.image_to_base64(image)
            payload = {
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "images": [image_base64],
                "seed": seed if seed > 0 else 0
            }

            response = requests.post(
                f"{baseurl}/v1/video/create",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            task_id = result.get("id")
                
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(30)

            max_attempts = 60  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/v1/video/query?id={task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")
                    
                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_result = status_response.json()
                    status = status_result.get("status", "")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status == "completed":
                        video_url = status_result.get("video_url")
                        break
                    elif status == "failed":
                        fail_reason = status_result.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        raise Exception(error_message)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(100)
            print(f"Video generation completed. URL: {video_url}")
            
            response_data = {
                "code": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "video_url": video_url,
            }
            
            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            raise Exception(error_message)


class Comfly_HaoeeImage_Gemini:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-3-pro-image-preview", "gemini-3-pro-image-preview（test）", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-image-preview（test）"], {"default": "gemini-3-pro-image-preview"}),
                "aspectRatio": (["auto", "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"], {"default": "auto"}),
                "imageSize": (["1K", "2K", "4K"], {"default": "1K"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 600
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=False)
    
    def generate_image(self, prompt, model="gemini-3-pro-image-preview", aspectRatio="auto", 
                      imageSize="1K", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # 正则匹配model是否包含（test）
            lineType = "main"
            if re.search(r'\（test\）', model, re.IGNORECASE):
                print(f"Test model detected: {model}")
                lineType = "test"
                model = re.sub(r'\（test\）', '', model, flags=re.IGNORECASE)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
                "lineType": lineType
            }

            all_images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
            base64_images = [self.image_to_base64(img) for img in all_images if img is not None]
            img_count = len(base64_images)
            print(f"Processing {img_count} input images")

            parts = [{ "text": f"{prompt},生成图片" }]
            if img_count > 0:
                parts.extend({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64
                    }
                } for b64 in base64_images)

            image_config = {
                "imageSize": imageSize
            }
            if aspectRatio != "auto":
                image_config["aspectRatio"] = aspectRatio

            payload = {
                "contents": [{
                    "role": "user",
                    "parts": parts
                }],
                "generationConfig": {
                    "responseModalities": ["TEXT","IMAGE"],
                    "imageConfig": image_config
                }
            }
            # parts = [{ "text": f"{prompt}" }]
            # if img_count > 0:
            #     parts.extend({"inline_data": {"mime_type": "image/png", "data": b64}} for b64 in base64_images)
                
            # payload = {
            #     "contents": [{'role': 'user', 'parts': parts }],
            #     "generationConfig": {
            #         "responseModalities": ["Image"],
            #         "imageConfig": {
            #             "aspectRatio": "" if aspectRatio == "auto" else aspectRatio,
            #             "imageSize": imageSize
            #         }
            #     },
            # }
            if seed > 0:
                payload["seed"] = seed

            api_model = model  # 已经去掉（test）
            url = f"{baseurl}/v1beta/models/{api_model}:generateContent"
            print(f"H: {headers}")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            pbar.update_absolute(30)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            candidates = result.get("candidates") or []
            content = candidates[0].get("content") if candidates else {}
            parts = content.get("parts") or []
            generated_tensors = []
            texts_only = []
            for part in parts:
                if "inlineData" in part:
                    image_base64 = part["inlineData"]["data"]
                    if image_base64:
                        image_data = base64.b64decode(image_base64)
                        generated_image = Image.open(BytesIO(image_data))
                        generated_tensor = pil2tensor(generated_image)
                        generated_tensors.append(generated_tensor)
                    # 文本处理
                elif "text" in part:
                    texts_only.append(part["text"])
                    
            response_info = f"Generated {len(generated_tensors)} images using {model}\n"
            if texts_only:
                response_info += "Text output:\n" + "\n".join(texts_only) + "\n" 
            else:
                response_info += f"imageSize: {imageSize}\n generated_tensors: {len(generated_tensors)}\n"
            pbar.update_absolute(100)
            print(f'generated_tensors: {len(generated_tensors)}')
            if generated_tensors:
                if len(generated_tensors) == 1:
                    combined_tensor = generated_tensors[0]  # 单张直接返回
                else:
                    combined_tensor = torch.cat(generated_tensors, dim=0)  # 多张拼接
                return (combined_tensor, response_info, "")
            else:
                # error_message = "Failed to process any images"
                # print(error_message)
                if texts_only:
                    raise Exception(response_info)
                else:
                    raise Exception("Failed to process any images or text")
                # return (None, error_message, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            traceback.print_exc()
            raise Exception(error_message)


class Comfly_HaoeeImage_Doubao_Seedream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seedream-4-5-251128", 'doubao-seedream-4-0-250828'], {"default": "doubao-seedream-4-5-251128"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "4:3", "3:4", "16:9", "9:16"], {"default": "1:1"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 300
        self.size_mapping = {
            "1K": {
                "1:1":  "1024x1024",
                "4:3":  "1024x768",
                "3:4":  "768x1024",
                "16:9": "1024x576",
                "9:16": "576x1024",
                "2:3":  "682x1024",
                "3:2":  "1024x682"
            },

            "2K": {
                "1:1":  "2048x2048",
                "4:3":  "2048x1536",
                "3:4":  "1536x2048",
                "16:9": "2560x1440",
                "9:16": "1440x2560",
                "2:3":  "1365x2048",
                "3:2":  "2048x1365"
            },

            "4K": {
                "1:1":  "4096x4096",
                "4:3":  "4096x3072",
                "3:4":  "3072x4096",
                "16:9": "4096x2304",
                "9:16": "2304x4096",
                "2:3":  "2731x4096",
                "3:2":  "4096x2731"
            }
        }

        self.resolution_factors = {
            "1K": 1,
            "2K": 2,
            "4K": 4
        }
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_image(self, prompt, model, response_format="url", resolution="1K", aspect_ratio="1:1", apikey="", 
                  image1=None, image2=None, image3=None, image4=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            if resolution in self.size_mapping and aspect_ratio in self.size_mapping[resolution]:
                final_size = self.size_mapping[resolution][aspect_ratio]
            else:
                final_size = "1024x1024"
                print(f"Warning: Combination of {resolution} resolution and {aspect_ratio} aspect ratio not found. Using {final_size}.")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            all_images = [image1, image2, image3, image4]
            base64_images = [self.image_to_base64(img) for img in all_images if img is not None]
            img_count = len(base64_images)

            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "size": final_size,
                "sequential_image_generation": "disabled",
                "watermark": False,
                "stream": False,
                "seed": seed if seed > 0 else 0
            }
            
            if img_count > 0:
                payload["image"] = base64_images
            
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            
            pbar.update_absolute(50)
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                raise Exception(error_message)
            
            image_url = None
            image_data = None
            generated_images = []
            image_urls = []
            for item in result["data"]:
                if response_format == "url":
                    image_url = item.get("url")
                    if not image_url:
                        continue
                    
                    image_urls.append(image_url)
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        image_data = BytesIO(img_response.content)
                        
                        pil_image = Image.open(image_data)
                        tensor_image = pil2tensor(pil_image)
                        generated_images.append(tensor_image)
                    except Exception as e:
                        print(f"Error downloading image: {str(e)}")
                else:
                    b64_data = item.get("b64_json")
                    if not b64_data:
                        continue
                        
                    image_data = BytesIO(base64.b64decode(b64_data))
                    
                    pil_image = Image.open(image_data)
                    tensor_image = pil2tensor(pil_image)
                    generated_images.append(tensor_image)
            
            pbar.update_absolute(80)
            if not generated_images:
                error_message = "Failed to process any images"
                print(error_message)
                raise Exception(error_message)
            
            combined_tensor = torch.cat(generated_images, dim=0)
                
            response_info = {
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "size": final_size,
                "seed": seed if seed != -1 else "auto",
                "urls": image_urls if image_urls else [],
                "aspect_ratio": aspect_ratio
            }

            if aspect_ratio == "Custom":
                response_info["original_dimensions"] = f"{width}x{height}"
                response_info["scaled_dimensions"] = final_size
            
            response_info["images_generated"] = len(generated_images)
            
            pbar.update_absolute(100)
            first_image_url = image_urls[0] if image_urls else ""
            return (combined_tensor, json.dumps(response_info, indent=2, ensure_ascii=False), first_image_url)
                
        except Exception as e:
            error_message = f"Error generating image: {str(e)}"
            print(error_message)
            raise Exception(error_message)


class Comfly_HaoeeImage_gpt_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "model": (["gpt-image-1.5", 'gpt-4o-image-vip'], {"default": "gpt-image-1.5"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 300

    def generate_image(self, prompt, model="gpt-image-1.5", n=1, quality="auto", 
                size="auto", background="auto", output_format="png", 
                moderation="auto", seed=0, api_key=""):
        if api_key.strip():
            self.api_key = api_key

        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }
            payload = {
                "prompt": prompt,
                "model": model,
                "n": n,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "moderation": moderation,
            }

            if size != "auto":
                payload["size"] = size
            if model == "gpt-image-1.5":
                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                pbar.update_absolute(30)
                print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")

                if response.status_code != 200:
                    error_message = f"API Error: {response.status_code} - {response.text}"
                    raise Exception(error_message)

                result = response.json()
                
                pbar.update_absolute(50)
            
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                response_info = f"**GPT-image-1 Generation ({timestamp})**\n\n"
                response_info += f"Prompt: {prompt}\n"
                response_info += f"Model: {model}\n"
                response_info += f"Quality: {quality}\n"
                if size != "auto":
                    response_info += f"Size: {size}\n"
                response_info += f"Background: {background}\n"
                response_info += f"Seed: {seed} (Note: Seed not used by API)\n\n"

                generated_images = []
                image_urls = []
                
                if "data" in result and result["data"]:
                    for i, item in enumerate(result["data"]):
                        pbar.update_absolute(50 + (i+1) * 50 // len(result["data"]))
                        
                        if "b64_json" in item:
                            b64_data = item["b64_json"]
                            if b64_data.startswith("data:image/png;base64,"):
                                b64_data = b64_data[len("data:image/png;base64,"):]    
                            image_data = base64.b64decode(b64_data)
                            generated_image = Image.open(BytesIO(image_data))
                            generated_tensor = pil2tensor(generated_image)
                            generated_images.append(generated_tensor)
                        elif "url" in item:
                            image_urls.append(item["url"])
                            try:
                                img_response = requests.get(item["url"])
                                if img_response.status_code == 200:
                                    generated_image = Image.open(BytesIO(img_response.content))
                                    generated_tensor = pil2tensor(generated_image)
                                    generated_images.append(generated_tensor)
                            except Exception as e:
                                print(f"Error downloading image from URL: {str(e)}")
                else:
                    error_message = "No generated images in response"
                    print(error_message)
                    response_info += f"Error: {error_message}\n"
                    raise Exception(response_info)

                if "usage" in result:
                    response_info += "Usage Information:\n"
                    if "total_tokens" in result["usage"]:
                        response_info += f"Total Tokens: {result['usage']['total_tokens']}\n"
                    if "input_tokens" in result["usage"]:
                        response_info += f"Input Tokens: {result['usage']['input_tokens']}\n"
                    if "output_tokens" in result["usage"]:
                        response_info += f"Output Tokens: {result['usage']['output_tokens']}\n"

                    if "input_tokens_details" in result["usage"]:
                        response_info += "Input Token Details:\n"
                        details = result["usage"]["input_tokens_details"]
                        if "text_tokens" in details:
                            response_info += f"  Text Tokens: {details['text_tokens']}\n"
                        if "image_tokens" in details:
                            response_info += f"  Image Tokens: {details['image_tokens']}\n"
                
                if generated_images:
                    combined_tensor = torch.cat(generated_images, dim=0)
                    
                    pbar.update_absolute(100)
                    return (combined_tensor, response_info)
                else:
                    error_message = "No images were successfully processed"
                    print(error_message)
                    response_info += f"Error: {error_message}\n"
                    raise Exception(response_info)
            else:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                response = requests.post(
                    f"{baseurl}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                print(f"Request sent to {response.url}. "
                    f"Response status code: {response.status_code}, "
                    f"Response text: {response.text}")

                if response.status_code != 200:
                    error_message = f"API Error: {response.status_code} - {response.text}"
                    print(error_message)
                    raise Exception(error_message)
                # ---------- 2. 解析返回 ----------
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                pbar.update_absolute(40)
                # ---------- 3. 提取图片 URL（重点修复） ----------
                image_urls = re.findall(
                    r"!\[.*?\]\((https?://[^)]+)\)",
                    content
                )

                if not image_urls:
                    error_message = "No image URLs found in response"
                    print(error_message)
                    raise Exception(content)

                # ---------- 4. 下载并转 IMAGE ----------
                generated_images = []

                for url in image_urls:
                    try:
                        img_resp = requests.get(url, timeout=60)
                        img_resp.raise_for_status()

                        img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                        img_tensor = pil2tensor(img)
                        generated_images.append(img_tensor)

                    except Exception as e:
                        print(f"[WARN] Failed to download image: {url} | {e}")

                if not generated_images:
                    error_message = "Images found but failed to download"
                    print(error_message)
                    raise Exception(content)
                # ---------- 5. 合并 batch ----------
                combined_tensor = torch.cat(generated_images, dim=0)
                pbar.update_absolute(100)
                # ---------- 6. 正确 RETURN ----------
                return (combined_tensor, content)

        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            raise Exception(error_message)


class Comfly_HaoeeImage_Midjourney:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "botType": (["MID_JOURNEY", "NIJI_JOURNEY"],{"default":"MID_JOURNEY"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "state": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 600
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_image(self, prompt, botType="MID_JOURNEY", image1=None, image2=None, image3=None, image4=None, state="", apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": "mj_imagine"
            }

            all_images = [image1, image2, image3, image4]
            base64_images = [self.image_to_base64(img) for img in all_images if img is not None]
            img_count = len(base64_images)
            print(f"Processing {img_count} input images")

            payload = {
                "prompt": prompt,
                "botType": botType,
                "base64Array": base64_images,
                "state": state,
                "seed": seed if seed > 0 else 0
            }
                        
            response = requests.post(
                f"{baseurl}/mj/submit/imagine",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            task_id = result.get("result")
                
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                raise Exception(error_message)
            
            pbar.update_absolute(40)

            max_attempts = 10  
            attempts = 0
            imageUrl = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    query_payload = {
                        "ids": [task_id]
                    }

                    status_response = requests.post(
                        f"{baseurl}/mj/task/list-by-condition",
                        headers=headers,
                        json=query_payload,
                        timeout=self.timeout
                    )
                    print(f"Request sent to {status_response.url}. Response status code: {status_response.status_code}, Response text: {status_response.text}")
                    
                    if status_response.status_code != 200:
                        error_message = f"Status check failed: {status_response.status_code} - {status_response.text}"
                        raise Exception(error_message)
                        
                    status_result = status_response.json()
                    status_data = status_result[0] if status_result else {}
                    status = status_data.get("status", "")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        imageUrl = status_data.get("imageUrl")
                        break
                    elif status == "FAILURE":
                        fail_reason = status_data.get("fail_reason", "Unknown error")
                        error_message = f"Image generation failed: {fail_reason}"
                        print(error_message)
                        raise Exception(error_message)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not imageUrl:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                raise Exception(error_message)
              

            try:
                img_response = requests.get(imageUrl, timeout=self.timeout)
                img_response.raise_for_status()
                image_data = BytesIO(img_response.content)
                
                pil_image = Image.open(image_data)
                tensor_image = pil2tensor(pil_image)
            except Exception as e:
                raise Exception(f"Error downloading image: {str(e)}")
                
            pbar.update_absolute(100)

            response_info = {
                "prompt": prompt,
                "botType": botType,
                "state": state,
                "seed": seed if seed != -1 else "auto",
                "imageUrl": imageUrl
            }

            return (tensor_image, json.dumps(response_info, ensure_ascii=False), "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            traceback.print_exc()
            raise Exception(error_message)


class Comfly_HaoeeImage_Nano_banana2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-3.1-flash-image-preview", "gemini-3.1-flash-image-preview（test）"], {"default": "gemini-3.1-flash-image-preview"}),
                "aspectRatio": (["auto", "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"], {"default": "auto"}),
                "imageSize": (["1K", "2K", "4K"], {"default": "1K"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 600
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=False)
    
    def generate_image(self, prompt, model="gemini-3.1-flash-image-preview", aspectRatio="auto", 
                      imageSize="1K", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found"
            print(error_message)
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # 正则匹配model是否包含（test）
            lineType = "main"
            if re.search(r'\（test\）', model, re.IGNORECASE):
                print(f"Test model detected: model")
                lineType = "test"
                model = re.sub(r'\（test\）', '', model, flags=re.IGNORECASE)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
                "lineType": lineType
            }

            all_images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
            base64_images = [self.image_to_base64(img) for img in all_images if img is not None]
            img_count = len(base64_images)
            print(f"Processing {img_count} input images")

            parts = [{ "text": f"{prompt}" }]
            if img_count > 0:
                parts.extend({"inline_data": {"mime_type": "image/png", "data": b64}} for b64 in base64_images)
                
            payload = {
                "contents": [{'role': 'user', 'parts': parts }],
                "generationConfig": {
                    "responseModalities": ["Image"],
                    "imageConfig": {
                        "aspectRatio": "" if aspectRatio == "auto" else aspectRatio,
                        "imageSize": imageSize
                    }
                },
                "seed": seed if seed > 0 else 0
            }
                        
            response = requests.post(
                f"{baseurl}/v1beta/models/gemini-3.1-flash-image-preview:generateContent",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}")
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                raise Exception(error_message)
                
            result = response.json()
            candidates = result.get("candidates") or []
            content = candidates[0].get("content") if candidates else {}
            parts = content.get("parts") or []
            
            generated_tensors = []
            for part in parts:
                if "inlineData" in part:
                    image_base64 = part["inlineData"]["data"]
                    if image_base64:
                        image_data = base64.b64decode(image_base64)
                        generated_image = Image.open(BytesIO(image_data))
                        generated_tensor = pil2tensor(generated_image)
                        generated_tensors.append(generated_tensor)
             
            response_info = f"Generated {len(generated_tensors)} images using {model}\n"
            response_info += f"imageSize: {imageSize}\n"       
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)
                return (combined_tensor, response_info, "")
            else:
                error_message = "Failed to process any images"
                print(error_message)
                raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            traceback.print_exc()
            raise Exception(error_message)


def _haoee_parse_images_payload(result, prompt, model, size, response_format, extra_headline="GPT Image 2 Generation"):
    log_prefix = "[HaoeeParseImages]"
    print(f"{log_prefix} ==> start: model={model}, size={size}, response_format={response_format}, "
          f"headline={extra_headline!r}, result_keys={list(result.keys()) if isinstance(result, dict) else type(result).__name__}")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    response_info = f"**{extra_headline} ({timestamp})**\n\n"
    response_info += f"Prompt: {prompt}\n"
    response_info += f"Model: {model}\n"
    if size:
        response_info += f"Size: {size}\n"
    if response_format:
        response_info += f"Response Format: {response_format}\n"
    response_info += "\n"

    generated_images = []
    data_items = result.get("data") or []
    print(f"{log_prefix} data_items count={len(data_items)}")
    if not data_items:
        print(f"{log_prefix} ERROR: no data in response, raw={json.dumps(result, ensure_ascii=False)[:500]}")
        raise Exception(f"No generated images in response: {json.dumps(result, ensure_ascii=False)}")

    for idx, item in enumerate(data_items):
        if "b64_json" in item and item["b64_json"]:
            b64_data = item["b64_json"]
            print(f"{log_prefix} item[{idx}] decode b64_json, len={len(b64_data)}")
            if b64_data.startswith("data:image/"):
                b64_data = b64_data.split(",", 1)[1]
            try:
                image_data = base64.b64decode(b64_data)
                generated_image = Image.open(BytesIO(image_data)).convert("RGB")
                print(f"{log_prefix} item[{idx}] decoded image size={generated_image.size}, bytes={len(image_data)}")
                generated_images.append(pil2tensor(generated_image))
            except Exception as e:
                print(f"{log_prefix} item[{idx}] ERROR decoding b64: {e}")
        elif "url" in item and item["url"]:
            url = item["url"]
            print(f"{log_prefix} item[{idx}] download url={url}")
            try:
                img_resp = requests.get(url, timeout=60)
                img_resp.raise_for_status()
                generated_image = Image.open(BytesIO(img_resp.content)).convert("RGB")
                print(f"{log_prefix} item[{idx}] downloaded image size={generated_image.size}, bytes={len(img_resp.content)}")
                generated_images.append(pil2tensor(generated_image))
                response_info += f"Image URL: {url}\n"
            except Exception as e:
                print(f"{log_prefix} item[{idx}] ERROR downloading {url}: {e}")
        else:
            print(f"{log_prefix} item[{idx}] skipped: no b64_json/url, keys={list(item.keys()) if isinstance(item, dict) else type(item).__name__}")

    if not generated_images:
        print(f"{log_prefix} ERROR: none of the {len(data_items)} items produced an image")
        raise Exception("Images found but failed to decode/download")

    if "usage" in result and result["usage"]:
        usage = result["usage"]
        print(f"{log_prefix} usage={json.dumps(usage, ensure_ascii=False)}")
        response_info += "\nUsage Information:\n"
        if "total_tokens" in usage:
            response_info += f"Total Tokens: {usage['total_tokens']}\n"
        if "input_tokens" in usage:
            response_info += f"Input Tokens: {usage['input_tokens']}\n"
        if "output_tokens" in usage:
            response_info += f"Output Tokens: {usage['output_tokens']}\n"
        details = usage.get("input_tokens_details") or {}
        if details:
            response_info += "Input Token Details:\n"
            if "text_tokens" in details:
                response_info += f"  Text Tokens: {details['text_tokens']}\n"
            if "image_tokens" in details:
                response_info += f"  Image Tokens: {details['image_tokens']}\n"

    combined_tensor = torch.cat(generated_images, dim=0)
    print(f"{log_prefix} <== done: generated_images={len(generated_images)}, tensor_shape={tuple(combined_tensor.shape)}")
    return combined_tensor, response_info


def _haoee_parse_results_payload(result, prompt, model, size):
    log_prefix = "[HaoeeParseResults]"
    print(f"{log_prefix} ==> start: model={model}, size={size}, "
          f"result_keys={list(result.keys()) if isinstance(result, dict) else type(result).__name__}")

    status = result.get("status")
    print(f"{log_prefix} status={status!r}")
    if status and status != "succeeded":
        reason = result.get("failure_reason") or result.get("error") or ""
        print(f"{log_prefix} ERROR: task not succeeded, reason={reason!r}")
        raise Exception(f"Task status={status}. {reason}".strip())

    items = result.get("results") or []
    print(f"{log_prefix} results count={len(items)}")
    if not items:
        print(f"{log_prefix} ERROR: empty results, raw={json.dumps(result, ensure_ascii=False)[:500]}")
        raise Exception(f"No results in response: {json.dumps(result, ensure_ascii=False)}")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    response_info = f"**GPT Image 2 Generation Test ({timestamp})**\n\n"
    response_info += f"Prompt: {prompt}\n"
    response_info += f"Model: {model}\n"
    if size:
        response_info += f"Size: {size}\n"
    task_id = result.get("id") or result.get("task_id")
    if task_id:
        response_info += f"Task ID: {task_id}\n"
        print(f"{log_prefix} task_id={task_id}")
    start_time = result.get("start_time")
    end_time = result.get("end_time")
    if start_time and end_time:
        response_info += f"Duration: {int(end_time) - int(start_time)}s\n"
        print(f"{log_prefix} duration={int(end_time) - int(start_time)}s (start={start_time}, end={end_time})")
    progress = result.get("progress")
    if progress is not None:
        response_info += f"Progress: {progress}\n"
    response_info += "\n"

    generated_images = []
    for idx, it in enumerate(items):
        url = it.get("url")
        if not url:
            print(f"{log_prefix} item[{idx}] skipped: no url, keys={list(it.keys()) if isinstance(it, dict) else type(it).__name__}")
            continue
        print(f"{log_prefix} item[{idx}] download url={url}")
        try:
            img_resp = requests.get(url, timeout=60)
            img_resp.raise_for_status()
            generated_image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            print(f"{log_prefix} item[{idx}] downloaded image size={generated_image.size}, bytes={len(img_resp.content)}")
            generated_images.append(pil2tensor(generated_image))
            response_info += f"Image URL: {url}\n"
        except Exception as e:
            print(f"{log_prefix} item[{idx}] ERROR downloading {url}: {e}")

    if not generated_images:
        print(f"{log_prefix} ERROR: {len(items)} results but no image downloaded")
        raise Exception("Results returned but failed to download any image")

    combined_tensor = torch.cat(generated_images, dim=0)
    print(f"{log_prefix} <== done: generated_images={len(generated_images)}, tensor_shape={tuple(combined_tensor.shape)}")
    return combined_tensor, response_info


def _haoee_safe_payload_for_log(payload, max_str_len=200):
    """Render a payload as JSON but shrink huge strings/lists (e.g. base64 images) for logging."""
    def shrink(v):
        if isinstance(v, str):
            return v if len(v) <= max_str_len else f"<str len={len(v)}>"
        if isinstance(v, list):
            if len(v) <= 10:
                return [shrink(x) for x in v]
            return f"<list len={len(v)}>"
        if isinstance(v, dict):
            return {k: shrink(vv) for k, vv in v.items()}
        return v
    try:
        return json.dumps(shrink(payload), ensure_ascii=False)
    except Exception as e:
        return f"<unprintable payload: {e}>"


def _haoee_safe_json_parse(response, log_prefix):
    """
    Parse response body as JSON, with clear diagnostics when:
      - body is empty
      - body is not valid JSON (HTML/plain text etc.)
    Raises Exception with a readable message; caller should let it bubble up.
    """
    body = response.text or ""
    content_type = response.headers.get("Content-Type", "")
    if not body.strip():
        msg = (f"Empty response body (status={response.status_code}, "
               f"content_type={content_type!r}, content_length={response.headers.get('Content-Length')})")
        print(f"{log_prefix} ERROR: {msg}")
        raise Exception(msg)
    try:
        return response.json()
    except Exception as e:
        preview = body if len(body) <= 500 else body[:500] + f"...<truncated, total_len={len(body)}>"
        msg = (f"Invalid JSON response (status={response.status_code}, "
               f"content_type={content_type!r}, parse_error={e}). body_preview={preview!r}")
        print(f"{log_prefix} ERROR: {msg}")
        raise Exception(msg)


class Comfly_HaoeeImage_Gpt_Image2_Generations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "size": (["1024x1024", "1536x1024", "1024x1536"], {"default": "1024x1024"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 300
        self.api_key = None

    def generate_image(self, prompt, model, size, api_key, response_format="b64_json",
                       image1=None, image2=None, image3=None, image4=None, seed=0):
        log_prefix = "[HaoeeGptImg2-Gen]"
        ref_count = sum(1 for x in [image1, image2, image3, image4] if x is not None)
        print(f"{log_prefix} ==> start: model={model}, size={size}, response_format={response_format}, "
              f"prompt_len={len(prompt)}, ref_images={ref_count}, seed={seed}")

        if api_key.strip():
            self.api_key = api_key
            print(f"{log_prefix} api_key overridden by input (len={len(api_key.strip())})")

        if not self.api_key:
            error_message = "API key not found"
            print(f"{log_prefix} ERROR: {error_message}")
            raise Exception(error_message)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
                "lineType": "main",
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "response_format": response_format,
            }

            refs = []
            for img in [image1, image2, image3, image4]:
                if img is not None:
                    refs.append(_image_tensor_to_base64(img, with_prefix=True))
            if refs:
                payload["image"] = refs
            print(f"{log_prefix} payload={_haoee_safe_payload_for_log(payload)}")

            pbar.update_absolute(25)
            request_url = f"{baseurl}/v1/images/generations"
            print(f"{log_prefix} POST {request_url}")
            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            print(f"{log_prefix} response.status_code={response.status_code}, url={response.url}, text_len={len(response.text)}")

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(f"{log_prefix} ERROR: non-200 response, body={response.text}")
                raise Exception(error_message)

            result = _haoee_safe_json_parse(response, log_prefix)
            pbar.update_absolute(60)

            combined_tensor, response_info = _haoee_parse_images_payload(
                result, prompt, model, size, response_format,
                extra_headline="GPT Image 2 Generation",
            )
            print(f"{log_prefix} parsed images_count={combined_tensor.shape[0] if hasattr(combined_tensor, 'shape') else 'n/a'}")
            pbar.update_absolute(100)
            print(f"{log_prefix} <== done")
            return (combined_tensor, response_info)

        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(f"{log_prefix} EXCEPTION: {e}")
            raise Exception(error_message)


class Comfly_HaoeeImage_Gpt_Image2_Generations_Test:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "size": ([
                    "auto",
                    "1:1", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4",
                    "21:9", "9:21", "1:3", "3:1", "2:1", "1:2",
                ], {"default": "auto"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 300
        self.api_key = None

    def generate_image(self, prompt, model, size, api_key,
                       image1=None, image2=None, image3=None, image4=None, seed=0):
        log_prefix = "[HaoeeGptImg2-GenTest]"
        ref_count = sum(1 for x in [image1, image2, image3, image4] if x is not None)
        print(f"{log_prefix} ==> start: model={model}, size={size}, "
              f"prompt_len={len(prompt)}, ref_images={ref_count}, seed={seed}")

        if api_key.strip():
            self.api_key = api_key
            print(f"{log_prefix} api_key overridden by input (len={len(api_key.strip())})")

        if not self.api_key:
            error_message = "API key not found"
            print(f"{log_prefix} ERROR: {error_message}")
            raise Exception(error_message)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
                "lineType": "test",
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "size": size,
            }

            refs = []
            for img in [image1, image2, image3, image4]:
                if img is not None:
                    refs.append(_image_tensor_to_base64(img, with_prefix=True))
            if refs:
                payload["urls"] = refs
            print(f"{log_prefix} payload={_haoee_safe_payload_for_log(payload)}")

            pbar.update_absolute(25)
            request_url = f"{baseurl}/v1/draw/completions"
            print(f"{log_prefix} POST {request_url} (SSE)")

            sse_headers = dict(headers)
            sse_headers["Accept"] = "text/event-stream"

            response = requests.post(
                request_url,
                headers=sse_headers,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            content_type = response.headers.get("Content-Type", "")
            print(f"{log_prefix} response.status_code={response.status_code}, url={response.url}, content_type={content_type!r}")

            if response.status_code != 200:
                try:
                    body_text = response.text
                except Exception:
                    body_text = "<unreadable>"
                print(f"{log_prefix} ERROR: non-200 response, body={body_text}")
                raise Exception(f"API Error: {response.status_code} - {body_text}")

            is_sse = ("text/event-stream" in content_type) or (not content_type)
            result = None

            if is_sse:
                print(f"{log_prefix} parsing SSE stream")
                last_progress = -1
                last_status = None
                event_count = 0

                for raw_line in response.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        print(f"{log_prefix} sse non-data line: {line[:120]}")
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        evt = json.loads(data_str)
                    except Exception as e:
                        print(f"{log_prefix} sse invalid JSON: {data_str[:200]!r}, err={e}")
                        continue

                    event_count += 1
                    progress = evt.get("progress")
                    status = evt.get("status")
                    failure_reason = (evt.get("failure_reason") or "").strip()
                    err = (evt.get("error") or "").strip()

                    status_changed = status != last_status
                    log_this = False
                    if status_changed:
                        log_this = True
                    elif isinstance(progress, int) and isinstance(last_progress, int) and progress - last_progress >= 10:
                        log_this = True
                    if log_this:
                        print(f"{log_prefix} sse event#{event_count}: status={status!r}, progress={progress}")
                        if status_changed:
                            print(f"{log_prefix} sse event#{event_count} full={json.dumps(evt, ensure_ascii=False)}")
                        last_status = status
                        if isinstance(progress, int):
                            last_progress = progress
                            mapped = 25 + int(progress * 0.7)
                            pbar.update_absolute(min(95, max(25, mapped)))

                    if failure_reason or err or status == "failed":
                        msg = failure_reason or err or "task failed"
                        print(f"{log_prefix} ERROR: sse reported failure: {msg}")
                        print(f"{log_prefix} ERROR: failure event full={json.dumps(evt, ensure_ascii=False)}")
                        raise Exception(f"Task failed: {msg}")

                    if status == "succeeded":
                        result = evt
                        print(f"{log_prefix} sse succeeded at event#{event_count}")
                        print(f"{log_prefix} sse succeeded event full={json.dumps(evt, ensure_ascii=False)}")
                        break

                print(f"{log_prefix} sse stream finished, total_events={event_count}")
                if result is None:
                    raise Exception(f"SSE stream ended without a succeeded event (events={event_count})")
            else:
                print(f"{log_prefix} content_type is not SSE, fallback to JSON parsing")
                result = _haoee_safe_json_parse(response, log_prefix)

            pbar.update_absolute(95)

            combined_tensor, response_info = _haoee_parse_results_payload(
                result, prompt, model, size,
            )
            print(f"{log_prefix} parsed images_count={combined_tensor.shape[0] if hasattr(combined_tensor, 'shape') else 'n/a'}")
            pbar.update_absolute(100)
            print(f"{log_prefix} <== done")
            return (combined_tensor, response_info)

        except Exception as e:
            error_message = f"Error in image generation (test): {str(e)}"
            print(f"{log_prefix} EXCEPTION: {e}")
            raise Exception(error_message)


class Comfly_HaoeeImage_Gpt_Image2_Edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "size": (["1024x1024", "1536x1024", "1024x1536"], {"default": "1024x1024"}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "edit_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = 300
        self.api_key = None

    def edit_image(self, image, prompt, model, api_key,
                   size="1024x1024", response_format="b64_json", seed=0):
        log_prefix = "[HaoeeGptImg2-Edit]"
        print(f"{log_prefix} ==> start: model={model}, size={size}, response_format={response_format}, "
              f"prompt_len={len(prompt)}, seed={seed}")

        if api_key.strip():
            self.api_key = api_key
            print(f"{log_prefix} api_key overridden by input (len={len(api_key.strip())})")

        if not self.api_key:
            error_message = "API key not found"
            print(f"{log_prefix} ERROR: {error_message}")
            raise Exception(error_message)

        if image is None:
            print(f"{log_prefix} ERROR: image not provided")
            raise Exception("Image not provided")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            pil_image = tensor2pil(image)[0]
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            buf.seek(0)
            image_bytes = buf.getvalue()
            buf.seek(0)
            print(f"{log_prefix} image prepared: size={pil_image.size}, png_bytes={len(image_bytes)}")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
                "lineType": "main",
            }

            data = {
                "prompt": prompt,
                "model": model,
                "size": size,
                "response_format": response_format,
            }

            files = {"image": ("image.png", buf, "image/png")}
            print(f"{log_prefix} form_data={_haoee_safe_payload_for_log(data)}, files=[image.png]")

            pbar.update_absolute(25)
            request_url = f"{baseurl}/v1/images/edits"
            print(f"{log_prefix} POST {request_url}")
            response = requests.post(
                request_url,
                headers=headers,
                data=data,
                files=files,
                timeout=self.timeout,
            )
            print(f"{log_prefix} response.status_code={response.status_code}, url={response.url}, text_len={len(response.text)}")

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(f"{log_prefix} ERROR: non-200 response, body={response.text}")
                raise Exception(error_message)

            result = _haoee_safe_json_parse(response, log_prefix)
            pbar.update_absolute(60)

            combined_tensor, response_info = _haoee_parse_images_payload(
                result, prompt, model, size, response_format,
                extra_headline="GPT Image 2 Edit",
            )
            print(f"{log_prefix} parsed images_count={combined_tensor.shape[0] if hasattr(combined_tensor, 'shape') else 'n/a'}")
            pbar.update_absolute(100)
            print(f"{log_prefix} <== done")
            return (combined_tensor, response_info)

        except Exception as e:
            error_message = f"Error in image edit: {str(e)}"
            print(f"{log_prefix} EXCEPTION: {e}")
            raise Exception(error_message)


class Comfly_HaoeeText:
    def __init__(self):
        self.timeout = 300

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "deepseek-r1",
                    "deepseek-v3.2",
                    "claude-opus-4-5-20251101",
                    "doubao-seed-1-8-251228",
                    "qwen3-max",
                    "qwen3-vl-plus",
                    "qwen-plus",
                    "glm-4.7",
                    "glm-4.7-flash",
                ], {"default": "deepseek-r1"}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "describe the image"}),
                "temperature": ("FLOAT", {"default": 0.6,"min": 0.0, "max": 2.0, "step": 0.1}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "describe")
    FUNCTION = "completions"
    CATEGORY = "好易/Text"

    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)

    def completions(self, apikey, model, role, prompt, temperature, seed=0, image1=None, image2=None, image3=None, 
                         image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, ):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found"
            print(error_message)
            return (error_message, "")
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
            }

            all_images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
            base64_images = [self.image_to_base64(img) for img in all_images if img is not None]
            img_count = len(base64_images)
            print(f"Processing {img_count} input images")


            content = [{'type': 'text', 'text': f"{prompt}"}]

            if img_count > 0:
                content.extend({'type': 'image_url', 'image_url': b64} for b64 in base64_images)
                
            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user', 'content': content },
            ]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed if seed > 0 else 0
            }
            
            response = requests.post(
                f"{baseurl}/v1/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )

            pbar.update_absolute(30)
            print(f"Request sent to {response.url}. Response status code: {response.status_code}, Response text: {response.text}")

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                return (error_message, "")
        
            result = response.json()
            pbar.update_absolute(40)

            if "error" in result:
                error_message = result["error"]
                print(error_message)
                return (error_message, "")

            if "choices" not in result or not result["choices"]:
                error_message = "No choices in response"
                print(error_message)
                return (error_message, "")
            
            prompt_result = result["choices"][0]["message"]["content"]

            if not prompt_result or not str(prompt_result).strip():
                error_message = "Empty response"
                print(error_message)
                return (error_message, "")

            response_info = {
                "prompt": prompt,
                "model": model,
                "img_count": img_count,
                "seed": seed if seed > 0 else 0
            }

            return (json.dumps(response_info, ensure_ascii=False), prompt_result)

        except Exception as e:
            error_message = f"Error completions: {str(e)}"
            print(error_message)
            return (error_message, "")


class Comfly_HaoeeTextGPT:

    def __init__(self):
        self.timeout = 300
        self.api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gpt-5.2",
                ], {"default": "gpt-5.2"}),

                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "describe the image"
                }),

                "temperature": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),

                "apikey": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "describe")
    FUNCTION = "completions"
    CATEGORY = "好易/Text"

    def completions(self, apikey, model, prompt, temperature):

        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            return ("API key not found", "")

        try:

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            payload = {
                "model": model,
                "input": prompt,
                "temperature": temperature
            }

            response = requests.post(
                f"{baseurl}/api/v2/openai/responses",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(30)

            print(f"Request sent to {response.url}")
            print(response.text)

            if response.status_code != 200:
                return (f"API Error: {response.status_code} - {response.text}", "")

            result = response.json()

            if result.get("error"):
                return (str(result["error"]), "")

            prompt_result = ""

            for item in result.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        prompt_result += c.get("text", "")

            if not prompt_result.strip():
                return ("Empty response", "")

            response_info = json.dumps({
                "model": model,
                "usage": result.get("usage", {}),
            }, ensure_ascii=False, indent=2)

            pbar.update_absolute(100)

            return (response_info, prompt_result)

        except Exception as e:
            return (f"Error completions: {str(e)}", "")


class Comfly_HaoeeTextGPT5_4:
    """
    好易 LLM GPT-5.4 节点。

    gpt-5.4 和 gpt-5.4-pro 请求体 + 响应体都不同：
      - gpt-5.4     : 请求 { model, messages, max_completion_tokens }
                      响应 Chat Completions 格式 choices[0].message.content
      - gpt-5.4-pro : 请求 { model, input }
                      响应 Responses API 格式 output[].content[].output_text
    解析时按字段优先级分派：choices > output，兼容两种格式。
    """

    def __init__(self):
        self.timeout = 600
        self.api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gpt-5.4",
                    "gpt-5.4-pro",
                ], {"default": "gpt-5.4"}),

                "prompt": ("STRING", { "multiline": True, "default": "" }),

                "max_completion_tokens": ("INT", {
                    "default": 300,
                    "min": 1,
                    "max": 131072,
                    "step": 1
                }),

                "apikey": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "describe")
    FUNCTION = "completions"
    CATEGORY = "好易/Text"

    def completions(self, apikey, model, prompt, max_completion_tokens):

        log_prefix = "[HaoeeGPT5.4]"
        print(f"{log_prefix} ==> start: model={model}, prompt_len={len(prompt)}, max_completion_tokens={max_completion_tokens}")

        if apikey.strip():
            self.api_key = apikey
            print(f"{log_prefix} apikey overridden by input (len={len(apikey.strip())})")

        if not self.api_key:
            print(f"{log_prefix} ERROR: API key not found")
            return ("API key not found", "")

        try:

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model
            }

            if model == "gpt-5.4-pro":
                payload = {
                    "model": model,
                    "input": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                print(f"{log_prefix} payload schema=responses-input (gpt-5.4-pro)")
            else:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_completion_tokens": max_completion_tokens
                }
                print(f"{log_prefix} payload schema=chat-completions-messages (gpt-5.4)")

            request_url = f"{baseurl}/v1/chat/completions"
            print(f"{log_prefix} POST {request_url}")
            print(f"{log_prefix} payload={json.dumps(payload, ensure_ascii=False)}")

            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(30)

            print(f"{log_prefix} response.status_code={response.status_code}, url={response.url}, text_len={len(response.text)}")
            print(f"{log_prefix} response.text={response.text}")

            if response.status_code != 200:
                print(f"{log_prefix} ERROR: non-200 response")
                return (f"API Error: {response.status_code} - {response.text}", "")

            result = response.json()

            if result.get("error"):
                print(f"{log_prefix} ERROR: result.error={result.get('error')}")
                return (str(result["error"]), "")

            prompt_result = ""
            parse_format = None

            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                parse_format = "chat_completions"
                for choice in choices:
                    msg = choice.get("message") if isinstance(choice, dict) else None
                    if not isinstance(msg, dict):
                        continue
                    content = msg.get("content")
                    if isinstance(content, str):
                        prompt_result += content
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and isinstance(c.get("text"), str):
                                prompt_result += c["text"]
            elif isinstance(result.get("output"), list):
                parse_format = "responses"
                for item in result.get("output") or []:
                    if not isinstance(item, dict):
                        continue
                    content_list = item.get("content") or []
                    if not isinstance(content_list, list):
                        continue
                    for c in content_list:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            text = c.get("text")
                            if isinstance(text, str):
                                prompt_result += text

            print(f"{log_prefix} parsed format={parse_format}, text_len={len(prompt_result)}")

            if not prompt_result.strip():
                print(f"{log_prefix} ERROR: empty text, raw_keys={list(result.keys())}")
                return ("Empty response", "")

            usage = result.get("usage", {}) or {}
            print(f"{log_prefix} usage={json.dumps(usage, ensure_ascii=False)}")

            response_info = json.dumps({
                "model": model,
                "usage": usage,
            }, ensure_ascii=False, indent=2)

            pbar.update_absolute(100)
            print(f"{log_prefix} <== done: model={model}")

            return (response_info, prompt_result)

        except Exception as e:
            print(f"{log_prefix} EXCEPTION: {e}")
            return (f"Error completions: {str(e)}", "")


NODE_CLASS_MAPPINGS = {
    "Comfly_Haoee_api_key": Comfly_Haoee_api_key,
    "Comfly_HaoeeVideo_MiniMax": Comfly_HaoeeVideo_MiniMax,
    "Comfly_HaoeeVideo_Sora2": Comfly_HaoeeVideo_Sora2,
    "Comfly_HaoeeVideo_Kling": Comfly_HaoeeVideo_Kling,
    # "Comfly_HaoeeVideo_vidu": Comfly_HaoeeVideo_vidu,
    # "Comfly_HaoeeVideo_Veo3": Comfly_HaoeeVideo_Veo3,
    "Comfly_HaoeeVideo_Wan": Comfly_HaoeeVideo_Wan,
    "Comfly_HaoeeVideo_Doubao": Comfly_HaoeeVideo_Doubao,
    "Comfly_HaoeeVideo_haoeedance": Comfly_HaoeeVideo_haoeedance,
    "Comfly_HaoeeImage_Gemini": Comfly_HaoeeImage_Gemini,
    "Comfly_HaoeeImage_Doubao_Seedream": Comfly_HaoeeImage_Doubao_Seedream,
    "Comfly_HaoeeImage_gpt_image": Comfly_HaoeeImage_gpt_image,
    # "Comfly_HaoeeImage_Gpt_Image2_Generations": Comfly_HaoeeImage_Gpt_Image2_Generations,
    "Comfly_HaoeeImage_Gpt_Image2_Generations_Test": Comfly_HaoeeImage_Gpt_Image2_Generations_Test,
    # "Comfly_HaoeeImage_Gpt_Image2_Edit": Comfly_HaoeeImage_Gpt_Image2_Edit,
    "Comfly_HaoeeImage_Midjourney": Comfly_HaoeeImage_Midjourney,
    # "Comfly_HaoeeImage_Nano_banana2": Comfly_HaoeeImage_Nano_banana2,
    "Comfly_HaoeeText": Comfly_HaoeeText,
    "Comfly_HaoeeTextGPT": Comfly_HaoeeTextGPT,
    "Comfly_HaoeeTextGPT5.4": Comfly_HaoeeTextGPT5_4,
}



NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfly_Haoee_api_key": "好易 API Key",
    "Comfly_HaoeeVideo_MiniMax": "好易 视频 MiniMax",
    "Comfly_HaoeeVideo_Sora2": "好易 视频 Sora2",
    "Comfly_HaoeeVideo_Kling": "好易 视频 Kling",
    # "Comfly_HaoeeVideo_vidu": "好易 视频 Vidu",
    # "Comfly_HaoeeVideo_Veo3": "好易 视频 Veo3",
    "Comfly_HaoeeVideo_Wan": "好易 视频 Wan",
    "Comfly_HaoeeVideo_Doubao": "好易 视频 Doubao",
    "Comfly_HaoeeVideo_haoeedance": "好易 视频 HaoeeDance",
    "Comfly_HaoeeImage_Gemini": "好易 绘图 Gemini",
    "Comfly_HaoeeImage_gpt_image": "好易 绘图 GPT Image",
    # "Comfly_HaoeeImage_Gpt_Image2_Generations": "好易 绘图 GPT Image2 图片生成",
    "Comfly_HaoeeImage_Gpt_Image2_Generations_Test": "好易 绘图 GPT Image2 图片生成(测试渠道)",
    # "Comfly_HaoeeImage_Gpt_Image2_Edit": "好易 绘图 GPT Image2 图片编辑",
    "Comfly_HaoeeImage_Doubao_Seedream": "好易 绘图 Doubao Seedream",
    "Comfly_HaoeeImage_Midjourney": "好易 绘图 Midjourney",
    # "Comfly_HaoeeImage_Nano_banana2": "好易 绘图 Nano banana2",
    "Comfly_HaoeeText": "好易 LLM",
    "Comfly_HaoeeTextGPT": "好易 LLM GPT",
    "Comfly_HaoeeTextGPT5.4": "好易 LLM GPT5.4",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']