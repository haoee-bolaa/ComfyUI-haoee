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
HAOEE_HTTP_TIMEOUT_SEC = 600  # 10 分钟：单次 HTTP 请求超时
HAOEE_POLL_TOTAL_TIMEOUT_SEC = 600  # 10 分钟：任务轮询总时长上限


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
                capture_output=True, text=True, timeout=HAOEE_HTTP_TIMEOUT_SEC
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
                response = requests.get(self.video_url, stream=True, timeout=HAOEE_HTTP_TIMEOUT_SEC)
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
    NODE_NAME = "ApiKey"

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
    NODE_NAME = "MiniMax"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, prompt, model="MiniMax-Hailuo-02", duration="6", resolution="768P", prompt_optimizer=True, image=None, api_key="", seed=0):
        if api_key.strip():
            self.api_key = api_key

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

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
            
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/api/v2/hailuo/v1/video_generation", 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("task_id")

            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task_id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            file_id = None
            video_url = None
            status_result = {}
            
            while time.monotonic() < poll_deadline:
                time.sleep(10)  
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/api/v2/get_task/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    state = status_result["data"]["state"]
                    
                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
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
                        fail_msg = status_result.get('base_resp', {}).get('status_msg', 'Unknown error')
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_msg}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            pbar.update_absolute(100)
            if not video_url:
                return (
                    None,
                    task_id,
                    json.dumps(status_result, ensure_ascii=False)
                )
            _haoee_log(self.NODE_NAME, f"done video_url={video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "file_id": file_id,
                "video_url": video_url,
            }
            
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeVideo_Sora2_Pro:
    NODE_NAME = "Sora2Pro"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["sora-2-pro"], {"default": "sora-2-pro"}),
                "seconds": (["4", "6", "8"], {"default": "4"}),
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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
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
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

        width, height = self.get_image_size(image)

        if (width, height) not in [(1280, 720), (720, 1280), (1024, 1792), (1792, 1024)]:
            _haoee_raise_local(self.NODE_NAME, f"image size must be one of 1280x720/720x1280/1024x1792/1792x1024, got {width}x{height}")

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
            _haoee_log_http_request(self.NODE_NAME, form_data, headers=headers, label="create", extra="files=[input_reference=image.png]")
            response = requests.post(
                f"{baseurl}/v1/videos",
                headers=headers,
                data=form_data,
                files=files,
                timeout=self.timeout
            )
            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("id")

            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None

            while time.monotonic() < poll_deadline:
                time.sleep(5)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/v1/videos/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_data = status_response.json()
                    status = status_data.get("status")

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    # status: queued, in_progress, completed, failed
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
                            _haoee_log(self.NODE_NAME, f"video saved: {file_path}")
                            video_url = file_path
                            break
                        # 如果是 JSON
                        else:
                            try:
                                content_data = content_response.json()
                                video_url = content_data.get("url", "")
                            except Exception:
                                video_url = ""

                            if video_url:
                                _haoee_log(self.NODE_NAME, f"video URL ready: {video_url}")
                                break
                            else:
                                _haoee_log(self.NODE_NAME, "content not ready, waiting 3s...")
                                time.sleep(3)
                    elif status == "failed":
                        err_obj = status_data.get("error") or {}
                        fail_reason = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_reason or 'Unknown error'}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video URL after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

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

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeVideo_Sora2:
    NODE_NAME = "Sora2"
    SORA2_CREATE_URL = f"{baseurl}/api/v1/generate_videos"
    SORA2_QUERY_URL = f"{baseurl}/api/v2/get_task/{{task_id}}"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["sora-2"], {"default": "sora-2"}),
                "duration_seconds": (["4", "8", "12"], {"default": "4"}),
                "resolution": (["720x1280", "1280x720", "1024x1792", "1792x1024"], {"default": "720x1280"}),
                "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {"default": "16:9"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "fps": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = None

    def _img_b64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)

    def generate_video(
        self,
        prompt,
        model,
        duration_seconds="4",
        resolution="720x1280",
        aspect_ratio="16:9",
        apikey="",
        image=None,
        negative_prompt="",
        fps="",
    ):
        if apikey and apikey.strip():
            self.api_key = apikey.strip()

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        mode = "image_to_video" if image is not None else "text_to_video"

        prompt_preview = (prompt[:80] + "...") if prompt and len(prompt) > 80 else (prompt or "")
        _haoee_log(
            self.NODE_NAME,
            f"call: type={mode}, model={model}, duration_seconds={duration_seconds}, "
            f"resolution={resolution}, aspect_ratio={aspect_ratio}, "
            f"image={image is not None}, negative_prompt={'yes' if negative_prompt and negative_prompt.strip() else 'no'}, "
            f"fps={fps!r}, prompt={prompt_preview!r}"
        )

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
            }

            config = {
                "duration_seconds": str(duration_seconds),
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            }
            if negative_prompt and negative_prompt.strip():
                config["negative_prompt"] = negative_prompt
            if fps and fps.strip():
                config["fps"] = fps
            if image is not None:
                config["reference_image_urls"] = [self._img_b64(image)]

            payload = {
                "type": mode,
                "channel": "openai",
                "model": model,
                "prompt": prompt,
                "config": config,
            }

            _haoee_log(self.NODE_NAME, f"POST {self.SORA2_CREATE_URL}")
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                self.SORA2_CREATE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            if result.get("code") != 0:
                _haoee_raise_api(self.NODE_NAME, f"create failed: code={result.get('code')}, message={result.get('message')}")

            data = result.get("data") or {}
            task_id = data.get("task_sn")
            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task_sn missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}, init state={data.get('task_state')}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None
            status_result = {}

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        self.SORA2_QUERY_URL.format(task_id=task_id),
                        headers=headers,
                        timeout=self.timeout,
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    status_data = status_result.get("data") or {}
                    state = (status_data.get("state") or "").lower()

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    _haoee_log(self.NODE_NAME, f"task {task_id} state={state} (attempt {attempts})")

                    if state == "success":
                        file_info = status_data.get("file_info") or []
                        if isinstance(file_info, list) and file_info:
                            first = file_info[0] or {}
                            video_url = first.get("file_url") or first.get("url")
                        if not video_url:
                            file_urls = status_data.get("file_urls") or []
                            if isinstance(file_urls, str):
                                video_url = file_urls
                            elif isinstance(file_urls, list) and file_urls:
                                video_url = file_urls[0]
                        if video_url:
                            _haoee_log(self.NODE_NAME, f"task {task_id} succeeded")
                            break
                        _haoee_raise_parse(self.NODE_NAME, "success but no video url in response", preview=str(status_result))
                    elif state in ("fail", "error"):
                        stat_desc = status_data.get("stat_desc") or status_result.get("message") or "Unknown error"
                        _haoee_raise_api(self.NODE_NAME, f"task {state}: {stat_desc}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"done task_id={task_id}, video_url={video_url}")

            video_adapter = safe_video_adapter(video_url)
            return (video_adapter, task_id, json.dumps(status_result, ensure_ascii=False))

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeVideo_Kling:
    NODE_NAME = "Kling"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC

    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=False)

    def generate_video(self, image, prompt, model, duration, api_key, negative_prompt="", seed=0, image_tail=None, **kwargs):
        if api_key.strip():
            self.api_key = api_key

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

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

            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
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
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            if result["code"] != 0:
                _haoee_raise_api(self.NODE_NAME, f"create failed: code={result.get('code')}, message={result.get('message')}")

            task_id = result["data"]["task_id"]

            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task_id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None

            while time.monotonic() < poll_deadline:
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
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_data = status_response.json()
                    status = status_data["data"]["task_status"]

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if status == "succeed":
                        video_url = status_data["data"]["task_result"]["videos"][0]["url"]
                        break

                    elif status == "failed":
                        fail_reason = status_data["data"].get("task_status_msg", "Unknown error")
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_reason}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

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

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeVideo_vidu:
    NODE_NAME = "Vidu"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, image, model="viduq2-pro", prompt="", api_key="", is_rec=False, duration=5, seed=0, resolution="720p", 
                      movement_amplitude="auto", bgm=False):
        
        if api_key.strip():
            self.api_key = api_key

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

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

            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/ent/v2/img2video",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("task_id")

            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task_id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/ent/v2/tasks/{task_id}/creations",
                        headers=headers,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    state = status_result.get("state", "")

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                _haoee_log(self.NODE_NAME, f"video url: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {err_code}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error (attempt {attempts}): {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"done video_url={video_url}")

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

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeVideo_Veo3:
    NODE_NAME = "Veo3"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

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

            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/v2/videos/generations",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("task_id")

            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task_id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/v2/videos/generations/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    status = status_result.get("status", "")

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        if "data" in status_result and "output" in status_result["data"]:
                            video_url = status_result["data"]["output"]
                            break
                    elif status == "FAILURE":
                        fail_reason = status_result.get("fail_reason", "Unknown error")
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_reason}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"done video_url={video_url}")

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

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")
        

class Comfly_HaoeeVideo_Wan:
    NODE_NAME = "Wan"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_video(self, model, prompt, negative_prompt, resolution="720P", duration="5", prompt_extend=False, shot_type="single", audio=False, watermark=False, apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

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

            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/api/v1/services/aigc/video-generation/video-synthesis",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("output", {}).get("task_id")

            if not task_id:
                fail_msg = result.get("message")
                if fail_msg:
                    _haoee_raise_api(self.NODE_NAME, f"create failed: {fail_msg}")
                _haoee_raise_parse(self.NODE_NAME, "task_id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/api/v1/tasks/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    status = status_result.get("output", {}).get("task_status")

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if status == "SUCCEEDED":
                        video_url = status_result.get("output", {}).get("video_url")
                        break
                    elif status == "FAILED":
                        fail_reason = status_result.get("output", {}).get("message", "Unknown error")
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_reason}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"done video_url={video_url}")

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

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


def safe_video_adapter(video_url=None):
    if not video_url:
        return None
    try:
        return ComflyVideoAdapter(video_url)
    except Exception as e:
        print(f"[VideoAdapter] fallback to empty video: {e}")
        return None


# ===== Unified node logging & error helpers =====
# 异常 message 统一格式: [<NodeName>][<LEVEL>] <message>
# 5 类 LEVEL:
#   LOCAL     节点本地校验/前置错误 (缺参数、本地 IO 等)
#   API_HTTP  接口 HTTP 非 200
#   API_ERR   接口业务错误 (HTTP 200 但 code/status/error.message 异常)
#   NETWORK   网络层异常 (timeout/DNS/连接拒绝)
#   PARSE     节点本地解析响应失败 (JSON 解析失败、字段缺失)


class HaoeeNodeError(Exception):
    """Haoee 节点已分类异常基类。

    外层 try/except 捕获到此类型时应原样 re-raise，避免在 message 上再加一层前缀。
    """


_HAOEE_MEDIA_FIELD_KEYS = frozenset({
    "b64_json", "data", "image", "images", "inlinedata", "inline_data", "filedata",
    "bytesbase64encoded", "first_frame_image", "base64array",
})


def _haoee_is_media_field_key(key):
    if not isinstance(key, str):
        return False
    kl = key.lower()
    if kl in _HAOEE_MEDIA_FIELD_KEYS:
        return True
    return "b64" in kl or (kl.endswith("image") and kl != "image_size")


_HAOEE_B64_PAYLOAD_PREFIXES = ("iVBOR", "/9j/", "UklGR", "R0lGOD")
_HAOEE_MIN_KNOWN_FIELD_B64_LEN = 200
_HAOEE_MIN_FALLBACK_B64_LEN = 500


def _haoee_extract_b64_payload(s):
    text = s.strip()
    if ";base64," in text and text.lower().startswith("data:"):
        return text.split(";base64,", 1)[1].strip()
    return text


def _haoee_base64_char_ratio(payload):
    if not payload:
        return 0.0
    valid = sum(
        1 for c in payload
        if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
    )
    return valid / len(payload)


def _haoee_try_b64decode(payload):
    try:
        p = payload.strip()
        if not p:
            return False
        base64.b64decode(p, validate=True)
        return True
    except Exception:
        return False


def _haoee_is_base64_string(s, field_key=None):
    if not isinstance(s, str) or not s:
        return False
    payload = _haoee_extract_b64_payload(s)

    if ";base64," in s and s.lower().startswith("data:"):
        if _haoee_try_b64decode(payload):
            return True

    for prefix in _HAOEE_B64_PAYLOAD_PREFIXES:
        if payload.startswith(prefix) and _haoee_try_b64decode(payload):
            return True

    if field_key and _haoee_is_media_field_key(field_key):
        if len(s) >= _HAOEE_MIN_KNOWN_FIELD_B64_LEN and _haoee_base64_char_ratio(payload) >= 0.95:
            if _haoee_try_b64decode(payload):
                return True

    if len(s) >= _HAOEE_MIN_FALLBACK_B64_LEN and _haoee_base64_char_ratio(payload) >= 0.95:
        if _haoee_try_b64decode(payload):
            return True

    return False


def _haoee_replace_base64_in_json(obj, field_key=None):
    has_b64 = False

    def walk(val, key=None):
        nonlocal has_b64
        if isinstance(val, str):
            if _haoee_is_base64_string(val, key):
                has_b64 = True
                return f"<base64 len={len(val)}>"
            return val
        if isinstance(val, list):
            return [walk(item, key) for item in val]
        if isinstance(val, dict):
            return {k: walk(v, k) for k, v in val.items()}
        return val

    return walk(obj, field_key), has_b64


def _haoee_prepare_response_log(text):
    if not text:
        return ""
    if os.environ.get("HAOEE_LOG_FULL_BODY") == "1":
        if len(text) <= 50000:
            return text
        return text[:50000] + f"...<truncated, total_len={len(text)}>"
    try:
        obj = json.loads(text)
    except Exception:
        return text
    new_obj, has_b64 = _haoee_replace_base64_in_json(obj)
    if not has_b64:
        return text
    return json.dumps(new_obj, ensure_ascii=False)


def _haoee_prepare_request_log(payload):
    if payload is None:
        return ""
    try:
        new_obj, _ = _haoee_replace_base64_in_json(payload)
        return json.dumps(new_obj, ensure_ascii=False)
    except Exception as e:
        return f"<unprintable payload: {e}>"


def _haoee_log_http_request(node, payload, headers=None, label="request", extra=""):
    parts = []
    if headers is not None:
        parts.append(f"headers={json.dumps(headers, ensure_ascii=False)}")
    parts.append(f"body={_haoee_prepare_request_log(payload)}")
    if extra:
        parts.append(extra)
    _haoee_log(node, f"{label} " + ", ".join(parts))


def _haoee_log(node, msg):
    print(f"[{node}] {msg}")


def _haoee_log_http_response(node, response, label="response", extra=""):
    body_text = response.text or ""
    safe_body = _haoee_prepare_response_log(body_text)
    msg = f"{label} status={response.status_code}, body={safe_body}"
    if extra:
        msg = f"{label} status={response.status_code}, {extra}, body={safe_body}"
    _haoee_log(node, msg)


def _haoee_raise_local(node, msg):
    full = f"[{node}][LOCAL] {msg}"
    print(full)
    raise HaoeeNodeError(full)


def _haoee_raise_http(node, response, hint=""):
    try:
        body = response.text or ""
    except Exception:
        body = "<unreadable>"
    if len(body) > 500:
        body = body[:500] + "...<truncated>"
    suffix = f" ({hint})" if hint else ""
    full = f"[{node}][API_HTTP] HTTP {response.status_code}{suffix} - {body}"
    print(full)
    raise HaoeeNodeError(full)


def _haoee_raise_api(node, msg):
    full = f"[{node}][API_ERR] {msg}"
    print(full)
    raise HaoeeNodeError(full)


def _haoee_raise_network(node, exc):
    full = f"[{node}][NETWORK] {type(exc).__name__}: {exc}"
    print(full)
    raise HaoeeNodeError(full)


def _haoee_raise_parse(node, msg, preview=""):
    full = f"[{node}][PARSE] {msg}"
    if preview:
        prev = preview if len(preview) <= 200 else (preview[:200] + "...<truncated>")
        full += f" preview={prev!r}"
    print(full)
    raise HaoeeNodeError(full)


class Comfly_HaoeeVideo_Doubao:
    NODE_NAME = "Doubao"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = None

    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)

    def generate_video(self, prompt, model, resolution="720p", duration=5, ratio="16:9", apikey="", image=None, seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

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

            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/volc/v1/contents/generations/tasks",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()

            task_id = result.get("id")
            video_url = result.get("content", {}).get("video_url")

            if video_url:
                pbar.update_absolute(100)
                _haoee_log(self.NODE_NAME, f"sync video_url={video_url}")
                video_adapter = safe_video_adapter(video_url)
                return (video_adapter, task_id, json.dumps(result, ensure_ascii=False))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted (async): {task_id}")
            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/volc/v1/contents/generations/tasks/{task_id}",
                        headers=headers,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    status = status_result.get("status", "").lower()
                    video_url = status_result.get("content", {}).get("video_url")

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if status in ["succeeded", "success"] and video_url:
                        _haoee_log(self.NODE_NAME, f"async video_url={video_url}")
                        break
                    elif status in ["failed", "failure"]:
                        fail_reason = status_result.get("fail_reason", "Unknown error")
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_reason}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

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

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeVideo_haoeedance:
    NODE_NAME = "Seedance"
    HAOEEDANCE_CREATE_URL = f"{baseurl}/api/v3/contents/generations/tasks"
    HAOEEDANCE_QUERY_URL = f"{baseurl}/api/v3/contents/generations/tasks/{{id}}"

    MODEL_MAP = {
        "Seedance-2-0": "haoeedance-2-0",
        "Seedance-2-0-fast": "haoeedance-2-0-fast",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (list(cls.MODEL_MAP.keys()), {"default": "Seedance-2-0"}),
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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = None

    def _img_b64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)

    def _empty_image(self):
        return torch.zeros(1, 1, 1, 3)

    def _download_last_frame(self, url):
        try:
            resp = requests.get(url, timeout=HAOEE_HTTP_TIMEOUT_SEC)
            resp.raise_for_status()
            pil_img = Image.open(BytesIO(resp.content))
            return pil2tensor(pil_img)
        except Exception as e:
            _haoee_log(self.NODE_NAME, f"download last_frame failed: {e}")
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
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        api_model = self.MODEL_MAP.get(model, model)

        prompt_preview = (prompt[:80] + "...") if prompt and len(prompt) > 80 else (prompt or "")
        _haoee_log(
            self.NODE_NAME,
            f"call: model={model}({api_model}), resolution={resolution}, duration={duration}, "
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
                "modelname": api_model,
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
                _haoee_raise_local(self.NODE_NAME, "content empty: need prompt or at least one image/video/audio")

            content_summary = [
                {"type": item["type"], "role": item.get("role", "")} for item in content
            ]
            _haoee_log(self.NODE_NAME, f"content items ({len(content)}): {content_summary}")

            payload = {
                "model": api_model,
                "content": content,
                "resolution": resolution,
                "duration": int(duration),
                "ratio": ratio,
                "generate_audio": bool(generate_audio),
                "watermark": bool(watermark),
                "return_last_frame": bool(return_last_frame),
            }

            _haoee_log(self.NODE_NAME, f"POST {self.HAOEEDANCE_CREATE_URL}")
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                self.HAOEEDANCE_CREATE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("id")
            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None
            last_frame_url = None
            status_result = {}

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        self.HAOEEDANCE_QUERY_URL.format(id=task_id),
                        headers=headers,
                        timeout=self.timeout,
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    task_status = (status_result.get("status") or "").lower()
                    content_resp = status_result.get("content") or {}

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if task_status == "succeeded":
                        video_url = content_resp.get("video_url")
                        last_frame_url = content_resp.get("last_frame_url")
                        if video_url:
                            _haoee_log(self.NODE_NAME, f"task {task_id} succeeded, last_frame_url={'yes' if last_frame_url else 'no'}")
                            break
                        _haoee_raise_parse(self.NODE_NAME, "success but no video_url in response", preview=str(status_result))
                    elif task_status in ("failed", "expired", "cancelled"):
                        err = status_result.get("error") or {}
                        err_msg = err.get("message") if isinstance(err, dict) else str(err)
                        _haoee_raise_api(self.NODE_NAME, f"task {task_status}: {err_msg or 'Unknown error'}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

            pbar.update_absolute(90)

            if return_last_frame and last_frame_url:
                _haoee_log(self.NODE_NAME, f"downloading last_frame: {last_frame_url}")
                last_frame_tensor = self._download_last_frame(last_frame_url)
            else:
                last_frame_tensor = self._empty_image()

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"done task_id={task_id}, video_url={video_url}")

            video_adapter = safe_video_adapter(video_url)
            return (video_adapter, last_frame_tensor, task_id, json.dumps(status_result, ensure_ascii=False))

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeVideo_Grok_Video_3:
    NODE_NAME = "Grok"

    PENDING_STATES = {"pending", "processing", "in_progress", "queued", "running"}
    SUCCESS_STATES = {"completed", "succeeded", "success"}
    FAILED_STATES = {"failed", "error"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["grok-video-3"], {"default": "grok-video-3"}),
                "aspect_ratio": (["2:3", "3:2", "1:1"], {"default": "2:3"}),
                "size": (["720P"], {"default": "720P"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "好易/Video"

    def __init__(self):
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC

    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)

    def _collect_images(self, image, image2, image3, image4, image5):
        images_list = []
        for tensor in (image, image2, image3, image4, image5):
            if tensor is None:
                continue
            images_list.append(self.image_to_base64(tensor))
        return images_list

    def generate_video(self, prompt, model="grok-video-3", aspect_ratio="2:3",
                       size="720P", apikey="", image=None,
                       image2=None, image3=None, image4=None, image5=None, seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        if image is None:
            _haoee_raise_local(self.NODE_NAME, "image not provided")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            create_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "ModelName": model,
            }
            query_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "modelName": model,
            }

            images_list = self._collect_images(image, image2, image3, image4, image5)
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "images": images_list,
                "seed": seed if seed > 0 else 0,
            }

            _haoee_log_http_request(self.NODE_NAME, payload, headers=create_headers, label="create")
            response = requests.post(
                f"{baseurl}/v1/video/create",
                headers=create_headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(20)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="create task")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("id")

            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task_id missing in create response", preview=str(result))

            pbar.update_absolute(30)
            _haoee_log(self.NODE_NAME, f"task submitted: {task_id}")

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            video_url = None
            status_result = {}

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/v1/video/query?id={task_id}",
                        headers=query_headers,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    status = str(status_result.get("status", "")).lower()

                    api_progress = status_result.get("progress")
                    if isinstance(api_progress, (int, float)) and 0 <= api_progress <= 100:
                        progress_value = int(30 + api_progress * 0.6)
                    else:
                        progress_value = min(90, 30 + int((time.monotonic() - poll_started) * 60 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if status in self.SUCCESS_STATES:
                        video_url = status_result.get("video_url")
                        if not video_url:
                            _haoee_raise_parse(
                                self.NODE_NAME,
                                "success but video_url missing in query response",
                                preview=str(status_result),
                            )
                        break
                    if status in self.FAILED_STATES:
                        fail_reason = (
                            status_result.get("error")
                            or status_result.get("fail_reason")
                            or status_result.get("message")
                            or "Unknown error"
                        )
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_reason}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not video_url:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get video url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"done video_url={video_url}")

            response_data = {
                "code": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "video_url": video_url,
            }
            for key in ("thumbnail_url", "enhanced_prompt", "progress",
                        "status_update_time", "completed_at"):
                if key in status_result and status_result[key] is not None:
                    response_data[key] = status_result[key]

            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, task_id, json.dumps(response_data, ensure_ascii=False))

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeImage_Gemini:
    NODE_NAME = "Gemini"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview","gemini-3-pro-image-preview-lite","gemini-3.1-flash-image-preview-lite"], {"default": "gemini-3-pro-image-preview"}),
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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=False)
    
    def generate_image(self, prompt, model="gemini-3-pro-image-preview", aspectRatio="auto", 
                      imageSize="1K", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

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
            _haoee_log(self.NODE_NAME, f"processing {img_count} input images")

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

            # api_model 用于拼接请求 URL：-lite 模型需去掉 -lite 后缀，header 仍传原始名称
            api_model = model[:-len("-lite")] if model.endswith("-lite") else model
            url = f"{baseurl}/v1beta/models/{api_model}:generateContent"
            _haoee_log(self.NODE_NAME, f"POST {url}")
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            pbar.update_absolute(30)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="generateContent")
            _haoee_log_http_response(self.NODE_NAME, response)

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
                elif "text" in part:
                    texts_only.append(part["text"])

            response_info = f"Generated {len(generated_tensors)} images using {model}\n"
            if texts_only:
                response_info += "Text output:\n" + "\n".join(texts_only) + "\n"
            else:
                response_info += f"imageSize: {imageSize}\n generated_tensors: {len(generated_tensors)}\n"
            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"generated_tensors={len(generated_tensors)}")
            if generated_tensors:
                if len(generated_tensors) == 1:
                    combined_tensor = generated_tensors[0]
                else:
                    combined_tensor = torch.cat(generated_tensors, dim=0)
                return (combined_tensor, response_info, "")
            else:
                if texts_only:
                    _haoee_raise_api(self.NODE_NAME, f"no image returned. text: {response_info}")
                else:
                    _haoee_raise_parse(self.NODE_NAME, "no image or text in response", preview=str(result))

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeImage_Doubao_Seedream:
    NODE_NAME = "Seedream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seedream-5-0-260128", "doubao-seedream-4-5-251128", "doubao-seedream-4-0-250828"], {"default": "doubao-seedream-5-0-260128"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "size": (["1K", "2K", "3K", "4K"], {"default": "2K"}),
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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_image(self, prompt, model, response_format="url", size="2K", apikey="",
                  image1=None, image2=None, image3=None, image4=None, seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "ModelName": model,
            }

            all_images = [image1, image2, image3, image4]
            base64_images = [self.image_to_base64(img) for img in all_images if img is not None]
            img_count = len(base64_images)

            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "size": size,
                "sequential_image_generation": "disabled",
                "watermark": False,
                "stream": False,
            }
            
            if img_count > 0:
                payload["image"] = base64_images
            
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/api/v3/images/generations",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="generations")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()

            pbar.update_absolute(50)

            if "data" not in result or not result["data"]:
                _haoee_raise_parse(self.NODE_NAME, "no image data in response", preview=str(result))

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
                        _haoee_log(self.NODE_NAME, f"download image failed: {e}")
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
                _haoee_raise_parse(self.NODE_NAME, "failed to decode any images from response", preview=str(result))

            combined_tensor = torch.cat(generated_images, dim=0)

            response_info = {
                "prompt": prompt,
                "model": model,
                "size": size,
                "urls": image_urls if image_urls else [],
                "images_generated": len(generated_images),
            }
            if result.get("usage"):
                response_info["usage"] = result["usage"]

            pbar.update_absolute(100)
            first_image_url = image_urls[0] if image_urls else ""
            return (combined_tensor, json.dumps(response_info, indent=2, ensure_ascii=False), first_image_url)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeImage_gpt_image:
    NODE_NAME = "GptImage"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC

    def generate_image(self, prompt, model="gpt-image-1.5", n=1, quality="auto", 
                size="auto", background="auto", output_format="png", 
                moderation="auto", seed=0, api_key=""):
        if api_key.strip():
            self.api_key = api_key

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

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
                _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                pbar.update_absolute(30)
                if response.status_code != 200:
                    _haoee_raise_http(self.NODE_NAME, response, hint="generations")
                _haoee_log_http_response(self.NODE_NAME, response, label="create")

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
                                img_response = requests.get(item["url"], timeout=HAOEE_HTTP_TIMEOUT_SEC)
                                if img_response.status_code == 200:
                                    generated_image = Image.open(BytesIO(img_response.content))
                                    generated_tensor = pil2tensor(generated_image)
                                    generated_images.append(generated_tensor)
                            except Exception as e:
                                _haoee_log(self.NODE_NAME, f"download image failed: {e}")
                else:
                    _haoee_raise_parse(self.NODE_NAME, "no generated images in response", preview=str(result))

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
                    _haoee_raise_parse(self.NODE_NAME, "no images successfully decoded", preview=str(result))
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
                _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
                response = requests.post(
                    f"{baseurl}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code != 200:
                    _haoee_raise_http(self.NODE_NAME, response, hint="chat/completions")
                _haoee_log_http_response(self.NODE_NAME, response, label="chat")
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                pbar.update_absolute(40)
                image_urls = re.findall(
                    r"!\[.*?\]\((https?://[^)]+)\)",
                    content
                )

                if not image_urls:
                    _haoee_raise_api(self.NODE_NAME, f"no image URLs in chat content: {content}")

                generated_images = []

                for url in image_urls:
                    try:
                        img_resp = requests.get(url, timeout=HAOEE_HTTP_TIMEOUT_SEC)
                        img_resp.raise_for_status()

                        img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                        img_tensor = pil2tensor(img)
                        generated_images.append(img_tensor)

                    except Exception as e:
                        _haoee_log(self.NODE_NAME, f"download image failed: {url} | {e}")

                if not generated_images:
                    _haoee_raise_parse(self.NODE_NAME, f"images found but failed to download. content: {content}")
                combined_tensor = torch.cat(generated_images, dim=0)
                pbar.update_absolute(100)
                return (combined_tensor, content)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeImage_Midjourney:
    NODE_NAME = "MJ"

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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=True)
    
    def generate_image(self, prompt, botType="MID_JOURNEY", image1=None, image2=None, image3=None, image4=None, state="", apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

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
            _haoee_log(self.NODE_NAME, f"processing {img_count} input images")

            payload = {
                "prompt": prompt,
                "botType": botType,
                "base64Array": base64_images,
                "state": state,
                "seed": seed if seed > 0 else 0
            }
                        
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/mj/submit/imagine",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(30)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="submit/imagine")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            task_id = result.get("result")

            if not task_id:
                _haoee_raise_parse(self.NODE_NAME, "task_id missing in create response", preview=str(result))

            pbar.update_absolute(40)

            poll_started = time.monotonic()
            poll_deadline = poll_started + HAOEE_POLL_TOTAL_TIMEOUT_SEC
            attempts = 0
            imageUrl = None

            while time.monotonic() < poll_deadline:
                time.sleep(10)
                attempts += 1

                try:
                    query_payload = {
                        "ids": [task_id]
                    }

                    _haoee_log_http_request(self.NODE_NAME, query_payload, headers=headers, label="query")
                    status_response = requests.post(
                        f"{baseurl}/mj/task/list-by-condition",
                        headers=headers,
                        json=query_payload,
                        timeout=self.timeout
                    )
                    if status_response.status_code != 200:
                        _haoee_raise_http(self.NODE_NAME, status_response, hint=f"poll #{attempts}")
                    _haoee_log_http_response(self.NODE_NAME, status_response, label=f"poll #{attempts}")

                    status_result = status_response.json()
                    status_data = status_result[0] if status_result else {}
                    status = status_data.get("status", "")

                    progress_value = min(80, 40 + int((time.monotonic() - poll_started) * 40 / HAOEE_POLL_TOTAL_TIMEOUT_SEC))
                    pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        imageUrl = status_data.get("imageUrl")
                        break
                    elif status == "FAILURE":
                        fail_reason = status_data.get("fail_reason", "Unknown error")
                        _haoee_raise_api(self.NODE_NAME, f"task failed: {fail_reason}")

                except requests.exceptions.RequestException as e:
                    _haoee_log(self.NODE_NAME, f"poll request error: {e}")

            if not imageUrl:
                _haoee_raise_parse(self.NODE_NAME, f"failed to get image url after {HAOEE_POLL_TOTAL_TIMEOUT_SEC}s poll timeout")


            try:
                img_response = requests.get(imageUrl, timeout=self.timeout)
                img_response.raise_for_status()
                image_data = BytesIO(img_response.content)

                pil_image = Image.open(image_data)
                tensor_image = pil2tensor(pil_image)
            except Exception as e:
                _haoee_raise_parse(self.NODE_NAME, f"error downloading image: {e}")

            pbar.update_absolute(100)

            response_info = {
                "prompt": prompt,
                "botType": botType,
                "state": state,
                "seed": seed if seed != -1 else "auto",
                "imageUrl": imageUrl
            }

            return (tensor_image, json.dumps(response_info, ensure_ascii=False), "")

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeImage_Nano_banana2:
    NODE_NAME = "Nano2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-3.1-flash-image-preview"], {"default": "gemini-3.1-flash-image-preview"}),
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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
    
    def image_to_base64(self, image_tensor):
        return _image_tensor_to_base64(image_tensor, with_prefix=False)
    
    def generate_image(self, prompt, model="gemini-3.1-flash-image-preview", aspectRatio="auto", 
                      imageSize="1K", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

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
            _haoee_log(self.NODE_NAME, f"processing {img_count} input images")

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
                        
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/v1beta/models/gemini-3.1-flash-image-preview:generateContent",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="generateContent")
            _haoee_log_http_response(self.NODE_NAME, response)

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
                _haoee_raise_parse(self.NODE_NAME, "no images in response", preview=str(result))

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


def _haoee_parse_images_payload(result, prompt, model, size, response_format, extra_headline="GPT Image 2 Generation", node="ParseImages"):
    log_prefix = f"[{node}]"
    print(f"{log_prefix} parse_images ==> start: model={model}, size={size}, response_format={response_format}, "
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
        _haoee_raise_parse(node, "no generated images in response", preview=json.dumps(result, ensure_ascii=False))

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
                img_resp = requests.get(url, timeout=HAOEE_HTTP_TIMEOUT_SEC)
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
        _haoee_raise_parse(node, f"images found but failed to decode/download (count={len(data_items)})")

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


def _haoee_parse_results_payload(result, prompt, model, size, node="ParseResults"):
    log_prefix = f"[{node}]"
    print(f"{log_prefix} parse_results ==> start: model={model}, size={size}, "
          f"result_keys={list(result.keys()) if isinstance(result, dict) else type(result).__name__}")

    status = result.get("status")
    print(f"{log_prefix} status={status!r}")
    if status and status != "succeeded":
        reason = result.get("failure_reason") or result.get("error") or ""
        _haoee_raise_api(node, f"task status={status}. {reason}".strip())

    items = result.get("results") or []
    print(f"{log_prefix} results count={len(items)}")
    if not items:
        _haoee_raise_parse(node, "no results in response", preview=json.dumps(result, ensure_ascii=False))

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
            img_resp = requests.get(url, timeout=HAOEE_HTTP_TIMEOUT_SEC)
            img_resp.raise_for_status()
            generated_image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            print(f"{log_prefix} item[{idx}] downloaded image size={generated_image.size}, bytes={len(img_resp.content)}")
            generated_images.append(pil2tensor(generated_image))
            response_info += f"Image URL: {url}\n"
        except Exception as e:
            print(f"{log_prefix} item[{idx}] ERROR downloading {url}: {e}")

    if not generated_images:
        _haoee_raise_parse(node, f"{len(items)} results but no image downloaded")

    combined_tensor = torch.cat(generated_images, dim=0)
    print(f"{log_prefix} <== done: generated_images={len(generated_images)}, tensor_shape={tuple(combined_tensor.shape)}")
    return combined_tensor, response_info


def _haoee_safe_json_parse(response, log_prefix, node="SafeJsonParse"):
    """
    Parse response body as JSON, with clear diagnostics.
    On failure raises HaoeeNodeError with [NODE][PARSE] prefix; caller should let it bubble up.
    """
    body = response.text or ""
    content_type = response.headers.get("Content-Type", "")
    if not body.strip():
        msg = (f"empty response body (status={response.status_code}, "
               f"content_type={content_type!r}, content_length={response.headers.get('Content-Length')})")
        print(f"{log_prefix} ERROR: {msg}")
        _haoee_raise_parse(node, msg)
    try:
        return response.json()
    except Exception as e:
        preview = body if len(body) <= 500 else body[:500] + f"...<truncated, total_len={len(body)}>"
        msg = (f"invalid JSON response (status={response.status_code}, "
               f"content_type={content_type!r}, parse_error={e})")
        print(f"{log_prefix} ERROR: {msg}, body_preview={preview!r}")
        _haoee_raise_parse(node, msg, preview=preview)


class Comfly_HaoeeImage_Gpt_Image2_Generations:
    NODE_NAME = "GptImg2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "size": ([
                    "1024x1024（1K 1:1）", "2048x2048（2K 1:1）", "2880x2880（4K 1:1）",
                    "1280x720（1K 16:9）", "2048x1152（2K 16:9）", "3840x2160（4K 16:9）",
                    "720x1280（1K 9:16）", "1152x2048（2K 9:16）", "2160x3840（4K 9:16）",
                    "1024x768（1K 4:3）", "2048x1536（2K 4:3）", "3264x2448（4K 4:3）",
                    "768x1024（1K 3:4）", "1536x2048（2K 3:4）", "2448x3264（4K 3:4）",
                    "1008x672（1K 3:2）", "2016x1344（2K 3:2）", "3504x2336（4K 3:2）",
                    "672x1008（1K 2:3）", "1344x2016（2K 2:3）", "2336x3504（4K 2:3）",
                    "1040x832（1K 5:4）", "2080x1664（2K 5:4）", "3200x2560（4K 5:4）",
                    "832x1040（1K 4:5）", "1664x2080（2K 4:5）", "2560x3200（4K 4:5）",
                    "1344x576（1K 21:9）", "2016x864（2K 21:9）", "3696x1584（4K 21:9）",
                ], {"default": "1024x1024（1K 1:1）"}),
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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = None

    def generate_image(self, prompt, model, size, api_key, response_format="b64_json",
                       image1=None, image2=None, image3=None, image4=None, seed=0):
        log_prefix = f"[{self.NODE_NAME}]"
        ref_count = sum(1 for x in [image1, image2, image3, image4] if x is not None)
        api_size = size.split("（", 1)[0].strip() if size else size
        _haoee_log(self.NODE_NAME, f"==> start: model={model}, size={size} (api={api_size}), response_format={response_format}, "
              f"prompt_len={len(prompt)}, ref_images={ref_count}, seed={seed}")

        if api_key.strip():
            self.api_key = api_key
            _haoee_log(self.NODE_NAME, f"api_key overridden by input (len={len(api_key.strip())})")

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "ModelName": model,
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "size": api_size,
                "response_format": response_format,
            }

            refs = []
            for img in [image1, image2, image3, image4]:
                if img is not None:
                    refs.append(_image_tensor_to_base64(img, with_prefix=True))
            if refs:
                payload["image"] = refs
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")

            pbar.update_absolute(25)
            request_url = f"{baseurl}/v1/images/generations"
            _haoee_log(self.NODE_NAME, f"POST {request_url}")
            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="generations")
            _haoee_log_http_response(self.NODE_NAME, response)

            result = _haoee_safe_json_parse(response, log_prefix, node=self.NODE_NAME)
            pbar.update_absolute(60)

            combined_tensor, response_info = _haoee_parse_images_payload(
                result, prompt, model, api_size, response_format,
                extra_headline="GPT Image 2 Generation",
                node=self.NODE_NAME,
            )
            _haoee_log(self.NODE_NAME, f"parsed images_count={combined_tensor.shape[0] if hasattr(combined_tensor, 'shape') else 'n/a'}")
            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, "<== done")
            return (combined_tensor, response_info)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeImage_Gpt_Image2_PerCount:
    NODE_NAME = "GptImg2PerCount"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "size": ([
                    "1024x1024（1K 1:1）", "2048x2048（2K 1:1）",
                    "1280x720（1K 16:9）", "2048x1152（2K 16:9）",
                    "720x1280（1K 9:16）", "1152x2048（2K 9:16）",
                    "1024x768（1K 4:3）", "2048x1536（2K 4:3）",
                    "768x1024（1K 3:4）", "1536x2048（2K 3:4）",
                    "1008x672（1K 3:2）", "2016x1344（2K 3:2）",
                    "672x1008（1K 2:3）", "1344x2016（2K 2:3）",
                    "1040x832（1K 5:4）", "2080x1664（2K 5:4）",
                    "832x1040（1K 4:5）", "1664x2080（2K 4:5）",
                    "1344x576（1K 21:9）", "2016x864（2K 21:9）",
                ], {"default": "1024x1024（1K 1:1）"}),
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
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = None

    def generate_image(self, prompt, model, size, api_key, response_format="b64_json",
                       image1=None, image2=None, image3=None, image4=None, seed=0):
        log_prefix = f"[{self.NODE_NAME}]"
        ref_count = sum(1 for x in [image1, image2, image3, image4] if x is not None)
        api_size = size.split("（", 1)[0].strip() if size else size
        _haoee_log(self.NODE_NAME, f"==> start: model={model}, size={size} (api={api_size}), response_format={response_format}, "
              f"prompt_len={len(prompt)}, ref_images={ref_count}, seed={seed}")

        if api_key.strip():
            self.api_key = api_key
            _haoee_log(self.NODE_NAME, f"api_key overridden by input (len={len(api_key.strip())})")

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "ModelName": "gpt-image-2-1k2k",
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "size": api_size,
                "response_format": response_format,
            }

            refs = []
            for img in [image1, image2, image3, image4]:
                if img is not None:
                    refs.append(_image_tensor_to_base64(img, with_prefix=True))
            if refs:
                payload["image"] = refs
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")

            pbar.update_absolute(25)
            request_url = f"{baseurl}/v1/images/generations"
            _haoee_log(self.NODE_NAME, f"POST {request_url}")
            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="generations")
            _haoee_log_http_response(self.NODE_NAME, response)

            result = _haoee_safe_json_parse(response, log_prefix, node=self.NODE_NAME)
            pbar.update_absolute(60)

            combined_tensor, response_info = _haoee_parse_images_payload(
                result, prompt, model, api_size, response_format,
                extra_headline="GPT Image 2 Generation (PerCount)",
                node=self.NODE_NAME,
            )
            _haoee_log(self.NODE_NAME, f"parsed images_count={combined_tensor.shape[0] if hasattr(combined_tensor, 'shape') else 'n/a'}")
            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, "<== done")
            return (combined_tensor, response_info)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeImage_Gpt_Image2_4K:
    NODE_NAME = "GptImg24K"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "size": ([
                    "1024x1024（1K 1:1）", "2048x2048（2K 1:1）", "2880x2880（4K 1:1）",
                    "1280x720（1K 16:9）", "2048x1152（2K 16:9）", "3840x2160（4K 16:9）",
                    "720x1280（1K 9:16）", "1152x2048（2K 9:16）", "2160x3840（4K 9:16）",
                    "1024x768（1K 4:3）", "2048x1536（2K 4:3）", "3264x2448（4K 4:3）",
                    "768x1024（1K 3:4）", "1536x2048（2K 3:4）", "2448x3264（4K 3:4）",
                    "1008x672（1K 3:2）", "2016x1344（2K 3:2）", "3504x2336（4K 3:2）",
                    "672x1008（1K 2:3）", "1344x2016（2K 2:3）", "2336x3504（4K 2:3）",
                    "1040x832（1K 5:4）", "2080x1664（2K 5:4）", "3200x2560（4K 5:4）",
                    "832x1040（1K 4:5）", "1664x2080（2K 4:5）", "2560x3200（4K 4:5）",
                    "1344x576（1K 21:9）", "2016x864（2K 21:9）", "3696x1584（4K 21:9）",
                ], {"default": "3840x2160（4K 16:9）"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "好易/Image"

    def __init__(self):
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = None

    def generate_image(self, prompt, model, size, api_key, image=None, seed=0):
        log_prefix = f"[{self.NODE_NAME}]"
        has_ref_image = image is not None
        api_size = size.split("（", 1)[0].strip() if size else size
        _haoee_log(self.NODE_NAME, f"==> start: model={model}, size={size} (api={api_size}), "
              f"prompt_len={len(prompt)}, has_ref_image={has_ref_image}, seed={seed}")

        if api_key.strip():
            self.api_key = api_key
            _haoee_log(self.NODE_NAME, f"api_key overridden by input (len={len(api_key.strip())})")

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "ModelName": "gpt-image-2-4k",
            }

            if has_ref_image:
                pil_image = tensor2pil(image)[0]
                png_buffer = BytesIO()
                pil_image.save(png_buffer, format="PNG")
                png_bytes = png_buffer.getvalue()

                data = {
                    "model": model,
                    "prompt": prompt,
                    "size": api_size,
                    "response_format": "url",
                }
                files = {
                    "image": ("image.png", png_bytes, "image/png"),
                }
                log_payload = {**data, "image": f"<png {len(png_bytes)} bytes>"}
                _haoee_log_http_request(self.NODE_NAME, log_payload, headers=headers, label="create")

                pbar.update_absolute(25)
                request_url = f"{baseurl}/v1/images/edits"
                _haoee_log(self.NODE_NAME, f"POST {request_url} (edits)")
                response = requests.post(
                    request_url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.timeout,
                )
                response_format = "url"
                hint = "edits"
            else:
                headers["Content-Type"] = "application/json"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "size": api_size,
                }
                _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")

                pbar.update_absolute(25)
                request_url = f"{baseurl}/v1/images/generations"
                _haoee_log(self.NODE_NAME, f"POST {request_url} (generations)")
                response = requests.post(
                    request_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response_format = ""
                hint = "generations"

            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint=hint)
            _haoee_log_http_response(self.NODE_NAME, response)

            result = _haoee_safe_json_parse(response, log_prefix, node=self.NODE_NAME)
            pbar.update_absolute(60)

            combined_tensor, response_info = _haoee_parse_images_payload(
                result, prompt, model, api_size, response_format,
                extra_headline="GPT Image 2 4K",
                node=self.NODE_NAME,
            )
            _haoee_log(self.NODE_NAME, f"parsed images_count={combined_tensor.shape[0] if hasattr(combined_tensor, 'shape') else 'n/a'}")
            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, "<== done")
            return (combined_tensor, response_info)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")



class Comfly_HaoeeText:
    """好易 LLM 节点，对接 Haoee 原生 v1/chat/completions 接口。"""
    NODE_NAME = "Text"

    def __init__(self):
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "deepseek-r1",
                    "deepseek-v3.2",
                    "claude-opus-4-5-20251101",
                    "doubao-seed-1-8-251228",
                    "doubao-seed-2-0-lite-260215",
                    "qwen3-max",
                    "qwen3-vl-plus",
                    "qwen-plus",
                    "glm-4.7",
                    "glm-4.7-flash",
                    "gemini-3.1-pro-preview",
                    "gemini-3.5-flash",
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

        if not getattr(self, "api_key", None):
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

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
            _haoee_log(self.NODE_NAME, f"processing {img_count} input images")


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

            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                f"{baseurl}/v1/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )

            pbar.update_absolute(30)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="chat/completions")
            _haoee_log_http_response(self.NODE_NAME, response, label="create")

            result = response.json()
            pbar.update_absolute(40)

            if "error" in result and result["error"]:
                _haoee_raise_api(self.NODE_NAME, f"chat error: {result['error']}")

            if "choices" not in result or not result["choices"]:
                _haoee_raise_parse(self.NODE_NAME, "no choices in response", preview=str(result))

            prompt_result = result["choices"][0]["message"]["content"]

            if not prompt_result or not str(prompt_result).strip():
                _haoee_raise_api(self.NODE_NAME, "empty response content")

            response_info = {
                "prompt": prompt,
                "model": model,
                "img_count": img_count,
                "seed": seed if seed > 0 else 0
            }

            return (json.dumps(response_info, ensure_ascii=False), prompt_result)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeTextGPT5:
    """
    好易 LLM GPT5 节点。

    支持 gpt-5.4 / gpt-5.5 / gpt-5.4-pro，请求体 + 响应体分两类：
      - gpt-5.4 / gpt-5.5 : 请求 { model, messages:[{role, content:[{type,text}]}], max_completion_tokens }
                            响应 Chat Completions 格式 choices[0].message.content
      - gpt-5.4-pro       : 请求 { model, input:[{role, content}] }
                            响应 Responses API 格式 output[].content[].output_text
    解析时按字段优先级分派：choices > output，兼容两种格式。
    """
    NODE_NAME = "TextGPT5"

    def __init__(self):
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gpt-5.4",
                    "gpt-5.4-pro",
                    "gpt-5.5",
                ], {"default": "gpt-5.5"}),

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

        _haoee_log(self.NODE_NAME, f"==> start: model={model}, prompt_len={len(prompt)}, max_completion_tokens={max_completion_tokens}")

        if apikey.strip():
            self.api_key = apikey
            _haoee_log(self.NODE_NAME, f"apikey overridden by input (len={len(apikey.strip())})")

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        try:

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "ModelName": model
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
                _haoee_log(self.NODE_NAME, "payload schema=responses-input (gpt-5.4-pro)")
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
                _haoee_log(self.NODE_NAME, "payload schema=chat-completions-messages (gpt-5.4/gpt-5.5)")

            request_url = f"{baseurl}/v1/chat/completions"
            _haoee_log(self.NODE_NAME, f"POST {request_url}")
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")

            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(30)

            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="chat/completions")
            _haoee_log_http_response(self.NODE_NAME, response)

            result = response.json()

            if result.get("error"):
                _haoee_raise_api(self.NODE_NAME, f"chat error: {result['error']}")

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

            _haoee_log(self.NODE_NAME, f"parsed format={parse_format}, text_len={len(prompt_result)}")

            if not prompt_result.strip():
                _haoee_raise_parse(self.NODE_NAME, "empty text in response", preview=str(result))

            usage = result.get("usage", {}) or {}
            _haoee_log(self.NODE_NAME, f"usage={json.dumps(usage, ensure_ascii=False)}")

            response_info = json.dumps({
                "model": model,
                "usage": usage,
            }, ensure_ascii=False, indent=2)

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"<== done: model={model}")

            return (response_info, prompt_result)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


class Comfly_HaoeeTextGemini:
    """好易 LLM Gemini 节点，对接 Gemini 原生 generateContent 接口。"""
    NODE_NAME = "TextGemini"

    def __init__(self):
        self.timeout = HAOEE_HTTP_TIMEOUT_SEC
        self.api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["gemini-3.1-pro-preview"], {"default": "gemini-3.1-pro-preview"}),
                "system": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "include_thoughts": ("BOOLEAN", {"default": True}),
                "thinking_budget": ("INT", {"default": 0, "min": 0, "max": 262144, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning")
    FUNCTION = "completions"
    CATEGORY = "好易/Text"

    def completions(self, apikey, model, system, prompt, temperature,
                    top_p=1.0, include_thoughts=True, thinking_budget=0):
        _haoee_log(self.NODE_NAME, f"==> start: model={model}, prompt_len={len(prompt)}, "
              f"temperature={temperature}, top_p={top_p}, include_thoughts={include_thoughts}, "
              f"thinking_budget={thinking_budget}")

        if apikey.strip():
            self.api_key = apikey
            _haoee_log(self.NODE_NAME, f"apikey overridden by input (len={len(apikey.strip())})")

        if not self.api_key:
            _haoee_raise_local(self.NODE_NAME, "API key not provided")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "ModelName": model,
            }

            thinking_config = {"includeThoughts": include_thoughts}
            if thinking_budget > 0:
                thinking_config["thinkingBudget"] = thinking_budget

            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}],
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "thinkingConfig": thinking_config,
                },
            }
            if system.strip():
                payload["systemInstruction"] = {"parts": [{"text": system}]}

            url = f"{baseurl}/v1beta/models/{model}:generateContent"
            _haoee_log(self.NODE_NAME, f"POST {url}")
            _haoee_log_http_request(self.NODE_NAME, payload, headers=headers, label="create")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            pbar.update_absolute(30)
            if response.status_code != 200:
                _haoee_raise_http(self.NODE_NAME, response, hint="generateContent")
            _haoee_log_http_response(self.NODE_NAME, response)

            result = response.json()
            pbar.update_absolute(60)

            candidates = result.get("candidates") or []
            content = candidates[0].get("content") if candidates else {}
            parts = content.get("parts") or []
            answers = []
            thoughts = []
            for part in parts:
                if part.get("thought"):
                    thoughts.append(part.get("text", ""))
                elif "text" in part:
                    answers.append(part["text"])

            response_text = "\n".join(answers).strip()
            reasoning_text = "\n".join(thoughts).strip()
            if not response_text and not reasoning_text:
                _haoee_raise_parse(self.NODE_NAME, "no text in response", preview=str(result))

            pbar.update_absolute(100)
            _haoee_log(self.NODE_NAME, f"<== done: response_len={len(response_text)}, reasoning_len={len(reasoning_text)}")
            return (response_text, reasoning_text)

        except HaoeeNodeError:
            raise
        except requests.exceptions.RequestException as e:
            _haoee_raise_network(self.NODE_NAME, e)
        except Exception as e:
            traceback.print_exc()
            _haoee_raise_local(self.NODE_NAME, f"unexpected: {type(e).__name__}: {e}")


NODE_CLASS_MAPPINGS = {
    "Comfly_Haoee_api_key": Comfly_Haoee_api_key,
    "Comfly_HaoeeVideo_MiniMax": Comfly_HaoeeVideo_MiniMax,
    # "Comfly_HaoeeVideo_Sora2_Pro": Comfly_HaoeeVideo_Sora2_Pro,
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
    "Comfly_HaoeeVideo_Grok_Video_3": Comfly_HaoeeVideo_Grok_Video_3,
    "Comfly_HaoeeImage_Gpt_Image2_Generations": Comfly_HaoeeImage_Gpt_Image2_Generations,
    "Comfly_HaoeeImage_Gpt_Image2_PerCount": Comfly_HaoeeImage_Gpt_Image2_PerCount,
    "Comfly_HaoeeImage_Gpt_Image2_4K": Comfly_HaoeeImage_Gpt_Image2_4K,
    "Comfly_HaoeeImage_Midjourney": Comfly_HaoeeImage_Midjourney,
    # "Comfly_HaoeeImage_Nano_banana2": Comfly_HaoeeImage_Nano_banana2,
    "Comfly_HaoeeText": Comfly_HaoeeText,
    "Comfly_HaoeeTextGPT5": Comfly_HaoeeTextGPT5,
    "Comfly_HaoeeTextGemini": Comfly_HaoeeTextGemini,
}



NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfly_Haoee_api_key": "好易 API Key",
    "Comfly_HaoeeVideo_MiniMax": "好易 视频 MiniMax",
    # "Comfly_HaoeeVideo_Sora2_Pro": "好易 视频 Sora2 Pro",
    "Comfly_HaoeeVideo_Sora2": "好易 视频 Sora2",
    "Comfly_HaoeeVideo_Kling": "好易 视频 Kling",
    # "Comfly_HaoeeVideo_vidu": "好易 视频 Vidu",
    # "Comfly_HaoeeVideo_Veo3": "好易 视频 Veo3",
    "Comfly_HaoeeVideo_Wan": "好易 视频 Wan",
    "Comfly_HaoeeVideo_Doubao": "好易 视频 Doubao",
    "Comfly_HaoeeVideo_haoeedance": "好易 视频 Seedance",
    "Comfly_HaoeeImage_Gemini": "好易 绘图 Gemini",
    "Comfly_HaoeeImage_gpt_image": "好易 绘图 GPT Image",
    "Comfly_HaoeeVideo_Grok_Video_3": "好易 视频 Grok Video 3",
    "Comfly_HaoeeImage_Gpt_Image2_Generations": "好易 绘图 GPT Image2（按token）",
    "Comfly_HaoeeImage_Gpt_Image2_PerCount": "好易 绘图 GPT Image2 2K（按次）",
    "Comfly_HaoeeImage_Gpt_Image2_4K": "好易 绘图 GPT Image2 4K（按次）",
    "Comfly_HaoeeImage_Doubao_Seedream": "好易 绘图 Doubao Seedream",
    "Comfly_HaoeeImage_Midjourney": "好易 绘图 Midjourney",
    # "Comfly_HaoeeImage_Nano_banana2": "好易 绘图 Nano banana2",
    "Comfly_HaoeeText": "好易 LLM",
    "Comfly_HaoeeTextGPT5": "好易 LLM GPT5",
    "Comfly_HaoeeTextGemini": "好易 LLM Gemini",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']