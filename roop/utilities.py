# roop/utilities.py

import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

import roop.globals

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'

# Monkey patch SSL for macOS
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context

def run_ffmpeg(args: List[str]) -> bool:
    """
    Execute an FFmpeg command with the specified arguments.
    
    Args:
        args (List[str]): List of FFmpeg arguments.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.output.decode()}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg.")
        return False

def detect_fps(target_path: str) -> float:
    """
    Detect the frames per second (FPS) of a video using ffprobe.
    
    Args:
        target_path (str): Path to the video file.
    
    Returns:
        float: FPS value, defaults to 30 if detection fails.
    """
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of',
        'default=noprint_wrappers=1:nokey=1', target_path
    ]
    try:
        output = subprocess.check_output(command).decode().strip().split('/')
        if len(output) == 2:
            numerator, denominator = map(int, output)
            return numerator / denominator
        return float(output[0]) if output else 30.0
    except (subprocess.CalledProcessError, ValueError, IndexError):
        print(f"Failed to detect FPS for {target_path}, defaulting to 30.")
        return 30.0

def extract_frames(target_path: str, fps: float = 30) -> bool:
    """
    Extract frames from a video with specified FPS.
    
    Args:
        target_path (str): Path to the video file.
        fps (float): Frames per second to extract.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    temp_frame_quality = roop.globals.temp_frame_quality if roop.globals.temp_frame_quality is not None else 0
    temp_frame_quality = max(2, min(31, temp_frame_quality * 31 // 100))  # Clamp to FFmpeg qscale range
    return run_ffmpeg([
        '-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality),
        '-pix_fmt', 'rgb24', '-vf', f'fps={fps}', 
        os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)
    ])

def create_video(target_path: str, fps: float = 30) -> bool:
    """
    Create a video from extracted frames with specified FPS and encoder.
    
    Args:
        target_path (str): Path to the video file.
        fps (float): Frames per second for the output video.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    output_video_quality = roop.globals.output_video_quality if roop.globals.output_video_quality is not None else 35
    output_video_quality = max(1, min(51, (output_video_quality + 1) * 51 // 100))  # Clamp to FFmpeg CRF range
    commands = [
        '-hwaccel', 'auto', '-r', str(fps), '-i', 
        os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format), 
        '-c:v', roop.globals.output_video_encoder
    ]
    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
    return run_ffmpeg(commands)

def restore_audio(target_path: str, output_path: str) -> None:
    """
    Restore audio from the target video to the output video.
    
    Args:
        target_path (str): Path to the original video.
        output_path (str): Path to the output video.
    """
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg([
        '-i', temp_output_path, '-i', target_path, '-c:v', 'copy', 
        '-map', '0:v:0', '-map', '1:a:0', '-y', output_path
    ])
    if not done:
        move_temp(target_path, output_path)

def get_temp_frame_paths(target_path: str) -> List[str]:
    """
    Get a list of temporary frame paths.
    
    Args:
        target_path (str): Path to the video file.
    
    Returns:
        List[str]: List of frame file paths.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(os.path.join(glob.escape(temp_directory_path), '*.' + roop.globals.temp_frame_format))

def get_temp_directory_path(target_path: str) -> str:
    """
    Get the temporary directory path for a target file.
    
    Args:
        target_path (str): Path to the video file.
    
    Returns:
        str: Temporary directory path.
    """
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)

def get_temp_output_path(target_path: str) -> str:
    """
    Get the temporary output path for a target file.
    
    Args:
        target_path (str): Path to the video file.
    
    Returns:
        str: Temporary output path.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)

def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    """
    Normalize the output path based on source and target.
    
    Args:
        source_path (str): Path to the source file.
        target_path (str): Path to the target file.
        output_path (str): Proposed output path.
    
    Returns:
        Optional[str]: Normalized output path, or None if invalid.
    """
    if source_path and target_path and output_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path

def create_temp(target_path: str) -> None:
    """
    Create a temporary directory for a target file.
    
    Args:
        target_path (str): Path to the video file.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)

def move_temp(target_path: str, output_path: str) -> None:
    """
    Move the temporary output to the final output path.
    
    Args:
        target_path (str): Path to the video file.
        output_path (str): Path to the final output file.
    """
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)

def clean_temp(target_path: str) -> None:
    """
    Clean up temporary files unless keep_frames is set.
    
    Args:
        target_path (str): Path to the video file.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)

def has_image_extension(image_path: str) -> bool:
    """
    Check if a path has a supported image extension.
    
    Args:
        image_path (str): Path to the file.
    
    Returns:
        bool: True if it has a supported image extension.
    """
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))

def is_image(image_path: str) -> bool:
    """
    Check if a path is a valid image file.
    
    Args:
        image_path (str): Path to the file.
    
    Returns:
        bool: True if it is a valid image.
    """
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False

def is_video(video_path: str) -> bool:
    """
    Check if a path is a valid video file.
    
    Args:
        video_path (str): Path to the file.
    
    Returns:
        bool: True if it is a valid video.
    """
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False

def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    """
    Download files from URLs if not already present, with progress bar.
    
    Args:
        download_directory_path (str): Directory to save downloaded files.
        urls (List[str]): List of URLs to download.
    """
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

def resolve_relative_path(path: str) -> str:
    """
    Resolve a relative path to an absolute path.
    
    Args:
        path (str): Relative path.
    
    Returns:
        str: Absolute path.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
