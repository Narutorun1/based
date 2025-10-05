import os
import sys
import importlib
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm
import cv2
import numpy as np
from gfpgan import GFPGANer  # For face enhancement

import roop

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_frames',
    'process_image',
    'process_video',
    'post_process'
]

# Global restorer for face enhancement (lazy init)
FACE_RESTORER = None

def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'roop.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError:
        sys.exit(f'Frame processor {frame_processor} not found.')
    except NotImplementedError:
        sys.exit(f'Frame processor {frame_processor} not implemented correctly.')
    return frame_processor_module

def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES

def initialize_restorer():
    global FACE_RESTORER
    if FACE_RESTORER is None and roop.globals.enhance_faces:
        FACE_RESTORER = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=1)

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None], erode_mask: float = 0.2, enhance_faces: bool = False) -> None:
    initialize_restorer()
    with ThreadPoolExecutor(max_workers=roop.globals.execution_threads) as executor:
        futures = []
        queue = create_queue(temp_frame_paths)
        queue_per_future = max(len(temp_frame_paths) // roop.globals.execution_threads, 1)
        while not queue.empty():
            future = executor.submit(process_frames_with_enhance, source_path, pick_queue(queue, queue_per_future), update, erode_mask, enhance_faces)
            futures.append(future)
        for future in as_completed(futures):
            future.result()

def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue

def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues

def process_frames_with_enhance(source_path: str, temp_frame_paths: List[str], update: Callable[[], None], erode_mask: float, enhance_faces: bool) -> None:
    for frame_path in temp_frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Apply mask erosion for smoother mouth edges
            if erode_mask > 0:
                mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
                kernel = np.ones((int(erode_mask * 100), int(erode_mask * 100)), np.uint8)
                eroded_mask = cv2.erode(mask, kernel, iterations=1)
                frame = cv2.bitwise_and(frame, frame, mask=eroded_mask)
            
            # Optional face enhancement for eye/mouth refinement
            if enhance_faces and FACE_RESTORER:
                _, _, restored = FACE_RESTORER.enhance(frame, has_aligned=False, only_center_face=True, pasted=False)
                if restored is not None:
                    frame = restored
            
            cv2.imwrite(frame_path, frame)
        update()

def process_video(source_path: str, frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        multi_process_frame(source_path, frame_paths, process_frames, lambda: update_progress(progress), roop.globals.erode_mask, roop.globals.enhance_faces)

def update_progress(progress: Any = None) -> None:
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    progress.set_postfix({
        'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
        'execution_providers': roop.globals.execution_providers,
        'execution_threads': roop.globals.execution_threads,
        'erode_mask': f'{roop.globals.erode_mask:.2f}' if hasattr(roop.globals, 'erode_mask') else '0.00',
        'enhance_faces': 'Yes' if roop.globals.enhance_faces else 'No'
    })
    progress.refresh()
    progress.update(1)
