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
from .utils import pil2tensor, tensor2pil
from comfy.comfy_types import IO

baseurl = "https://maas.haoee.com"


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, prompt, model="MiniMax-Hailuo-02", duration="6", resolution="768P", prompt_optimizer=True, image=None, api_key="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "错误，未配置api_key"}
            return (None, "", json.dumps(error_response, ensure_ascii=False))

        if image is None:
            error_message = "错误，未配置image"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                
            result = response.json()
            task_id = result.get("task_id")

            if not task_id:
                error_message = "错误，未获取到task_id"
                print(error_message)
                return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                        
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                    
                except Exception as e:
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
            import traceback
            traceback.print_exc()
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
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
            return (None, "", json.dumps(error_response, ensure_ascii=False))
            
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))

        width, height = self.get_image_size(image)

        if (width, height) not in [(1280, 720), (720, 1280), (1024, 1792), (1792, 1024)]:
            error_message = "图片尺寸必须为 1280x720, 720x1280, 1024x1792, or 1792x1024"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))

        if model == "sora-2" and size not in ["720x1280", "1280x720"]:
            error_message = "sora-2模型只支持720x1280和1280x720尺寸"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
        
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
                return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                
            result = response.json()
            task_id = result.get("id")
            
            if not task_id:
                error_message = "No task ID in API response"
                print(error_message)
                return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))
                        
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
                                waited += 3
                    elif status == "failed":
                        fail_reason = status_data.get("error", {}).get("message", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        return (None, task_id, json.dumps({"status": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))
                        
                except Exception as e:
                    print(f"Error checking task status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to get video URL after {max_attempts} attempts"
                print(error_message)
                return (None, task_id, json.dumps({"status": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))
            
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
            import traceback
            traceback.print_exc()
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))


class Comfly_HaoeeVideo_Kling:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_tail": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["kling-video-o1", "kling-v2-6", "kling-video-v2-5-turbo", "kling-v2-1-master"], {"default": "kling-v2-6"}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "mode": (["std", "pro"],{"default": "std"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"],{"default": "auto"}),
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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8') # kling的image不加base64前缀

    def generate_video(self, image, prompt, model, duration, resolution, api_key, negative_prompt="", seed=0,  image_tail=None, **kwargs):
        if api_key.strip():
            self.api_key = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return (None, "", json.dumps(error_response, ensure_ascii=False))

        if image is None:
            error_message = "Image not provided"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))

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
                 "kling-v2-1-master":  "kling-v2-1-master"
            }

            payload = {
                "model_name": model_map.get(model, model),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "duration": duration,
            }
            
            mode = kwargs.get("mode", "std")
            aspect_ratio = kwargs.get("aspect_ratio", "auto")
            sound = kwargs.get("sound", "off")

            if model == "kling-video-o1":
                payload.update({
                    "image_list": [{"image_url": image_base64}],
                    "mode": mode,
                    "aspect_ratio": aspect_ratio,
                    "sound": sound
                })
            else:
                payload["image"] = image_base64

                if model != "kling-v2-6":
                    payload["resolution"] = resolution

                if model in ["kling-v2-6","kling-video-v2-5-turbo"]:
                    payload["mode"] = mode

                if model == "kling-v2-6":
                    payload["sound"] = sound
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
                return (None, "", json.dumps(error_response, ensure_ascii=False))
            
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return (None, "", json.dumps(error_response, ensure_ascii=False))
                
            task_id = result["data"]["task_id"]
            
            if not task_id:
                error_message = "No task ID in API response"
                print(error_message)
                return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                        
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))

                except Exception as e:
                    print(f"Error checking task status: {str(e)}")

            if not video_url:
                error_message = f"Failed to get video URL after {max_attempts} attempts"
                print(error_message)
                return (None, task_id, json.dumps({"status": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))
            
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
            return (None, "", json.dumps(error_response, ensure_ascii=False))


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, image, model="viduq2-pro", prompt="", api_key="", is_rec=False, duration=5, seed=0, resolution="720p", 
                      movement_amplitude="auto", bgm=False):
        
        if api_key.strip():
            self.api_key = api_key
            
        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found"}
            return (None, "", json.dumps(error_response, ensure_ascii=False))
        
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                
            result = response.json()
            task_id = result.get("task_id")
            
            if not task_id:
                error_message = "No task ID in API response"
                print(error_message)
                return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                        
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
            import traceback
            traceback.print_exc()
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return (None, "", json.dumps(error_response, ensure_ascii=False))
    
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                return (None, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
                
            result = response.json()
            task_id = result.get("task_id")
                
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                return (None, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
            
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                        
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
                        return (None, task_id, json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
            return (None, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
        

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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, model, prompt, negative_prompt, resolution="720P", duration="5", prompt_extend=False, shot_type="single", audio=False, watermark=False, apikey="", image=None, seed=0):
        empty_video = None
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return (empty_video, "", json.dumps(error_response, ensure_ascii=False))
    
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            return (empty_video, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))

        if model == "wan2.6-i2v" and not audio:
            error_message = "wan2.6-i2v模型只能有声"
            print(error_message)
            return (empty_video, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                    "negative_prompt": negative_prompt,
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
                return (empty_video, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
                
            result = response.json()
            task_id = result.get("output", {}).get("task_id")
                
            if not task_id:
                error_message = result.get("message", "No task ID returned from API") 
                print(error_message)
                return (empty_video, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
            
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
                        return (empty_video, task_id, json.dumps({"status": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))
                        
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
                        return (empty_video, task_id, json.dumps({"code": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return (empty_video, task_id, json.dumps({"status": "error", "message": error_message, "task_id": task_id}, ensure_ascii=False))
            
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
            return (empty_video, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
   
        
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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"

    def generate_video(self, prompt, model, resolution="720p", duration=5, ratio="16:9", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return (None, "", json.dumps(error_response, ensure_ascii=False))

        if image is None:
            error_message = "Image not provided"
            print(error_message)
            return (None, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))

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
                return (None, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))

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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
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
                        return (None, task_id, json.dumps({"code": "error", "message": f"Video generation failed: {fail_reason}"}, ensure_ascii=False))

                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")

            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))

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
            return (None, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def generate_video(self, prompt, model="grok-video-3", aspect_ratio="2:3", size="720P", apikey="", image=None, seed=0):
        empty_video = None
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return (empty_video, "", json.dumps(error_response, ensure_ascii=False))
    
        if image is None:
            error_message = "Image not provided"
            print(error_message)
            return (empty_video, "", json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
                return (empty_video, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
                
            result = response.json()
            task_id = result.get("id")
                
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                return (empty_video, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
            
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
                        return (empty_video, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                        
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
                        return (empty_video, task_id, json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return (empty_video, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
            
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
            return (empty_video, "", json.dumps({"code": "error", "message": error_message}, ensure_ascii=False))


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image(self, prompt, model="gemini-3-pro-image-preview", aspectRatio="auto", 
                      imageSize="1K", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found"
            print(error_message)
            return (None, error_message, "")
            
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
                return (None, error_message, "")
                
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
                response_info += f"imageSize: {imageSize}\n"
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
                    return (None, response_info, "")
                else:
                    return (None, "Failed to process any images or text", "")
                # return (None, error_message, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (None, error_message, "")


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
        """Convert tensor to base64 string"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_image(self, prompt, model, response_format="url", resolution="1K", aspect_ratio="1:1", apikey="", 
                  image1=None, image2=None, image3=None, image4=None, seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            return (None, error_message, "")
            
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
                return (None, error_message, "")
                
            result = response.json()
            
            pbar.update_absolute(50)
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                return (None, error_message, "")
            
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
                return (None, error_message, "")
            
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
            return (None, error_message, "")


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
            return (None, error_message)
            
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
                    return (None, error_message)

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
                    return (None, response_info)

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
                    return (None, response_info)
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
                    return (None, error_message)
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
                    return (None, content)

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
                    return (None, content)
                # ---------- 5. 合并 batch ----------
                combined_tensor = torch.cat(generated_images, dim=0)
                pbar.update_absolute(100)
                # ---------- 6. 正确 RETURN ----------
                return (combined_tensor, content)

        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            return (None, error_message)


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def generate_image(self, prompt, botType="MID_JOURNEY", image1=None, image2=None, image3=None, image4=None, state="", apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found"
            print(error_message)
            return (None, error_message, "")
            
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
                return (None, error_message, "")
                
            result = response.json()
            task_id = result.get("result")
                
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                return (None, error_message, "")
            
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
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                        
                    status_result = status_response.json()
                    status_data = status_result[0] if status_result else {}
                    status = status_data.get("status", "")

                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        imageUrl = status_data.get("imageUrl")
                        break
                    elif status == "FAILURE":
                        fail_reason = status_result.get("fail_reason", "Unknown error")
                        error_message = f"Image generation failed: {fail_reason}"
                        print(error_message)
                        return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
                    
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not imageUrl:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return (None, task_id, json.dumps({"status": "error", "message": error_message}, ensure_ascii=False))
              

            try:
                img_response = requests.get(imageUrl, timeout=self.timeout)
                img_response.raise_for_status()
                image_data = BytesIO(img_response.content)
                
                pil_image = Image.open(image_data)
                tensor_image = pil2tensor(pil_image)
            except Exception as e:
                print(f"Error downloading image: {str(e)}")
                
            pbar.update_absolute(100)

            response_info = {
                "prompt": prompt,
                "botType": botType,
                "state": state,
                "seed": seed if seed != -1 else "auto",
                "imageUrl": imageUrl
            }

            return (tensor_image, response_info, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (None, error_message, "")


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
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image(self, prompt, model="gemini-3.1-flash-image-preview", aspectRatio="auto", 
                      imageSize="1K", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey
            
        if not self.api_key:
            error_message = "API key not found"
            print(error_message)
            return (None, error_message, "")
            
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
                return (None, error_message, "")
                
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
                return (None, error_message, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (None, error_message, "")


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
                    "gemini-3-pro-preview",
                    "glm-4.7",
                    "glm-4.7-flash",
                ], {"default": "gemini-3-pro-preview"}),
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
        """Convert tensor to base64 string"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"

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

            return (json.dumps(response_info, ensure_ascii=False), prompt_result)

        except Exception as e:
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
    "Comfly_HaoeeImage_Gemini": Comfly_HaoeeImage_Gemini,
    "Comfly_HaoeeImage_Doubao_Seedream": Comfly_HaoeeImage_Doubao_Seedream,
    "Comfly_HaoeeImage_gpt_image": Comfly_HaoeeImage_gpt_image,
    "Comfly_HaoeeImage_Midjourney": Comfly_HaoeeImage_Midjourney,
    # "Comfly_HaoeeImage_Nano_banana2": Comfly_HaoeeImage_Nano_banana2,
    "Comfly_HaoeeText": Comfly_HaoeeText,
    "Comfly_HaoeeTextGPT": Comfly_HaoeeTextGPT,
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
    "Comfly_HaoeeImage_Gemini": "好易 绘图 Gemini",
    "Comfly_HaoeeImage_gpt_image": "好易 绘图 GPT Image",
    "Comfly_HaoeeImage_Doubao_Seedream": "好易 绘图 Doubao Seedream",
    "Comfly_HaoeeImage_Midjourney": "好易 绘图 Midjourney",
    # "Comfly_HaoeeImage_Nano_banana2": "好易 绘图 Nano banana2",
    "Comfly_HaoeeText": "好易 LLM",
    "Comfly_HaoeeTextGPT": "好易 LLM GPT",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']