"""Reolink camera client with ANPR capabilities."""

import asyncio
import re
import time
import cv2
import numpy as np
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple
from reolink_aio.api import Host
from .logger import logger

# RTSP frame streaming for ANPR
STREAM_FPS = 8  # Frames per second handed to ALPR (the camera itself sends ~25)
STREAM_CONNECT_TIMEOUT = 8  # Seconds to wait for the first frame


class CameraClient:
    """Reolink camera client for vehicle detection and ANPR."""

    def __init__(self, config):
        self.config = config
        self.host = None
        self.connected = False
        self.connection_time = 0
        self.processing_callback = None
        self.motion_detected_flag = False
        self.last_motion_time = 0
        self.processing_lock = None  # Will be created in connect()
        self.last_processed_time = 0
        self.is_processing = False

    async def connect(self) -> bool:
        """Connect to Reolink camera."""
        try:
            self.host = Host(
                self.config.camera_host,
                self.config.camera_username,
                self.config.camera_password
            )

            await self.host.login()
            await self.host.get_host_data()
            self.connected = True
            self.connection_time = time.time()

            # Create processing lock
            self.processing_lock = asyncio.Lock()
            
            # Ensure RTSP is enabled
            await self._ensure_rtsp_enabled()

            # Set up TCP push events via Baichuan protocol
            logger.info("Setting up TCP push events (Baichuan protocol)...")
            self.host.baichuan.register_callback(
                "anpr_motion_detection",
                self._motion_callback
            )
            await self.host.baichuan.subscribe_events()
            logger.info("TCP push events subscribed successfully")

            logger.info(f"Connected to camera at {self.config.camera_host}")
            logger.info(f"Monitoring: {self.config.camera_name} (Channel {self.config.camera_channel})")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return False

    def set_processing_callback(self, callback):
        """Set callback function to trigger when vehicle is detected."""
        self.processing_callback = callback

    def _motion_callback(self, *args, **kwargs):
        """Callback for motion events from TCP push."""
        current_time = time.time()

        # Ignore initialization events (first 3 seconds)
        if current_time - self.connection_time < 3.0:
            logger.debug("Ignoring initialization TCP push event")
            return

        logger.info(f"TCP push event received - Args: {args}, Kwargs: {kwargs}")

        # Set motion detection flags
        self.motion_detected_flag = True
        self.last_motion_time = current_time

        # Trigger callback for each TCP event (AI state filtering will happen in async task)
        if self.processing_callback:
            try:
                asyncio.create_task(self._check_and_process())
                logger.debug("Motion event callback triggered")
            except Exception as e:
                logger.error(f"Error triggering callback: {e}")

    async def _check_and_process(self):
        """Check AI state and process if vehicle detected."""
        try:
            # Check AI state IMMEDIATELY to see if vehicle
            ai_state_started = time.monotonic()
            ai_state = await self.host.get_ai_state(self.config.camera_channel)
            logger.info(f"AI State: {ai_state} (took {time.monotonic() - ai_state_started:.2f}s)")

            if ai_state and ai_state.get('vehicle') == True:
                logger.info("VEHICLE motion confirmed via AI state!")

                # Process with lock (lock prevents simultaneous processing)
                if self.processing_lock:
                    # Try to acquire lock without waiting if already locked
                    # This prevents queue buildup during processing
                    if self.processing_lock.locked():
                        logger.debug("Processing already in progress - skipping this event")
                        return
                    
                    async with self.processing_lock:
                        await self.processing_callback()
                else:
                    await self.processing_callback()
            else:
                # Not a vehicle - just log
                motion_types = []
                if ai_state:
                    if ai_state.get('person'): motion_types.append('person')
                    if ai_state.get('face'): motion_types.append('face')
                    if ai_state.get('pet'): motion_types.append('pet')

                if motion_types:
                    logger.debug(f"Non-vehicle motion ignored: {', '.join(motion_types)}")
                else:
                    logger.debug("Motion event but no AI state match")
        except Exception as e:
            logger.error(f"Error checking AI state: {e}")

    async def disconnect(self):
        """Disconnect from camera."""
        if self.host and self.connected:
            try:
                await self.host.baichuan.unsubscribe_events()
            except Exception as e:
                logger.debug(f"Error unsubscribing from events: {e}")

            await self.host.logout()
            self.connected = False
            logger.info("Disconnected from camera")

    async def get_isp_settings(self) -> Optional[Dict]:
        """Get current camera ISP settings."""
        try:
            body = [{"cmd": "GetIsp", "action": 1, "param": {"channel": self.config.camera_channel}}]
            result = await self.host.send(body)

            if result and len(result) > 0 and result[0].get("code") == 0:
                return result[0].get("value", {}).get("Isp", {})
            return None

        except Exception as e:
            logger.error(f"Failed to get ISP settings: {e}")
            return None

    async def set_isp_settings(self, settings: Dict) -> bool:
        """Set camera ISP settings."""
        try:
            # Log incoming settings request
            logger.info(f"Requested ISP settings change: {settings}")
            
            # Build the ISP parameter object - ONLY send what we want to change
            isp_data = {
                "channel": self.config.camera_channel,
            }
            
            # Add the settings we want to change
            for key, value in settings.items():
                isp_data[key] = value
            
            # Handle exposure mode logic for Auto mode
            exposure_mode = settings.get("exposure")
            if exposure_mode == "Auto":
                # For Auto mode, ensure we send full ranges for gain/shutter
                # But only if they're not already specified in settings
                if "gain" not in settings:
                    isp_data["gain"] = {"min": 1, "max": 100}
                if "shutter" not in settings:
                    isp_data["shutter"] = {"min": 0, "max": 125}
            
            isp_param = {"Isp": isp_data}
            body = [{"cmd": "SetIsp", "action": 0, "param": isp_param}]
            
            logger.info(f"Sending to camera API: {isp_param}")
            result = await self.host.send(body)
            logger.info(f"Camera API response: {result}")

            if result and len(result) > 0 and result[0].get("code") == 0:
                logger.info(f"ISP settings applied successfully")
                
                # Verify by reading back
                await asyncio.sleep(0.5)  # Small delay for camera to apply
                verify = await self.get_isp_settings()
                if verify:
                    logger.info(f"Verified settings after apply - dayNight: {verify.get('dayNight')}, exposure: {verify.get('exposure')}, binningMode: {verify.get('binningMode')}, nr3d: {verify.get('nr3d')}")
                
                return True
            else:
                logger.warning(f"ISP settings may have failed. Response: {result}")
                return False

        except Exception as e:
            logger.error(f"Failed to set ISP settings: {e}")
            return False

    async def apply_recording_settings(self, mode: str) -> bool:
        """Apply before/after recording settings from config."""
        try:
            # Get settings from config
            if mode == 'before' and hasattr(self.config, 'before_recording_settings'):
                settings = self.config.before_recording_settings
            elif mode == 'after' and hasattr(self.config, 'after_recording_settings'):
                settings = self.config.after_recording_settings
            else:
                return False  # No settings configured

            if not settings:
                return False  # Empty settings

            # Log what we're applying
            logger.debug(f"{mode.capitalize()} settings: {settings}")

            # Apply settings with single retry
            for attempt in range(2):
                success = await self.set_isp_settings(settings)
                if success:
                    return True
                if attempt == 0:  # Retry once
                    await asyncio.sleep(0.3)

            logger.error(f"❌ Failed to apply {mode}-recording settings")
            return False

        except Exception as e:
            logger.error(f"Error applying {mode}-recording settings: {e}")
            return False

    def _mask_credentials(self, text: str) -> str:
        """Hide the camera password in text destined for logs (e.g. FFmpeg's RTSP URL echo)."""
        text = re.sub(r"(rtsp://[^:/@\s]+:)[^@\s]*@", r"\1***@", text)
        password = self.config.camera_password
        return text.replace(password, "***") if password else text

    async def stream_frames(self, duration_seconds: float) -> AsyncIterator[bytes]:
        """Yield JPEG frames straight from the RTSP stream, always the newest one.

        Frames that arrive while the consumer is busy are dropped, so processing never
        works on stale images. The scan window starts at the first frame and lasts
        duration_seconds; closing the iterator (e.g. on early exit) stops ffmpeg.
        """
        # Reolink uses 1-based channel numbering
        rtsp_url = f"rtsp://{self.config.camera_username}:{self.config.camera_password}@{self.config.camera_host}:554/h264Preview_{self.config.camera_channel+1:02d}_main"

        ffmpeg_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-rtsp_transport', 'tcp',
            '-fflags', 'nobuffer', '-flags', 'low_delay',  # Don't wait to fill input buffers
            '-probesize', '32768', '-analyzeduration', '0',  # Skip long stream probing
            '-i', rtsp_url,
            '-an',
            '-vf', f'fps={STREAM_FPS}',
            '-c:v', 'mjpeg', '-q:v', '3',
            '-f', 'image2pipe', 'pipe:1',
        ]

        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        latest: Optional[bytes] = None
        new_frame = asyncio.Event()
        finished = False
        stderr_chunks: List[bytes] = []

        async def read_frames():
            nonlocal latest, finished
            buffer = bytearray()
            try:
                while True:
                    chunk = await process.stdout.read(262144)
                    if not chunk:
                        break
                    buffer += chunk
                    # Each JPEG starts with FFD8 and ends with FFD9
                    while True:
                        start = buffer.find(b'\xff\xd8')
                        end = buffer.find(b'\xff\xd9', start + 2) if start >= 0 else -1
                        if start < 0 or end < 0:
                            break
                        latest = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]
                        new_frame.set()
            finally:
                finished = True
                new_frame.set()

        async def read_stderr():
            data = await process.stderr.read()
            if data:
                stderr_chunks.append(data)

        reader = asyncio.create_task(read_frames())
        stderr_reader = asyncio.create_task(read_stderr())

        try:
            deadline = started + STREAM_CONNECT_TIMEOUT
            first_frame_at = None
            while True:
                if first_frame_at is not None and time.monotonic() >= deadline:
                    break
                try:
                    await asyncio.wait_for(new_frame.wait(), timeout=max(deadline - time.monotonic(), 0.01))
                except asyncio.TimeoutError:
                    if first_frame_at is None:
                        logger.error(f"No frame from RTSP stream within {STREAM_CONNECT_TIMEOUT}s")
                    break
                new_frame.clear()
                if latest is None:
                    if finished:
                        break  # ffmpeg exited and everything was consumed
                    continue

                frame, latest = latest, None
                if first_frame_at is None:
                    first_frame_at = time.monotonic()
                    deadline = first_frame_at + duration_seconds
                    logger.info(f"First RTSP frame after {first_frame_at - started:.2f}s")
                yield frame
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
            await asyncio.gather(reader, stderr_reader, return_exceptions=True)
            if stderr_chunks and first_frame_at is None:
                logger.error(f"FFmpeg stderr: {self._mask_credentials(stderr_chunks[0].decode(errors='replace'))}")

    async def get_snapshot(self) -> Optional[bytes]:
        """Get a single snapshot from the camera."""
        try:
            snapshot = await self.host.get_snapshot(
                self.config.camera_channel,
                "main"
            )
            return snapshot
        except Exception as e:
            logger.error(f"Error getting snapshot: {e}")
            return None

    async def get_device_info(self) -> Dict:
        """Get camera device information."""
        try:
            return {
                'name': self.config.camera_name,
                'host': self.config.camera_host,
                'channel': self.config.camera_channel,
                'connected': self.connected
            }
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {}
    
    async def _ensure_rtsp_enabled(self):
        """Check and enable RTSP if it's disabled."""
        try:
            # Get current network port settings
            body = [{"cmd": "GetNetPort", "action": 0, "param": {"channel": 0}}]
            response = await self.host.send(body)
            
            if response and len(response) > 0:
                net_port = response[0].get("value", {}).get("NetPort", {})
                rtsp_enabled = net_port.get("rtspEnable", 0)
                rtsp_port = net_port.get("rtspPort", 554)
                
                if rtsp_enabled == 0:
                    logger.warning("RTSP is disabled - attempting to enable...")
                    
                    # Enable RTSP
                    enable_body = [{
                        "cmd": "SetNetPort",
                        "action": 0,
                        "param": {
                            "NetPort": {
                                "rtspEnable": 1,
                                "rtspPort": 554
                            }
                        }
                    }]
                    
                    enable_response = await self.host.send(enable_body)
                    
                    if enable_response and enable_response[0].get("code") == 0:
                        logger.info("✅ RTSP enabled successfully on port 554")
                    else:
                        logger.error(f"Failed to enable RTSP: {enable_response}")
                else:
                    logger.info(f"✅ RTSP already enabled on port {rtsp_port}")
                    
        except Exception as e:
            logger.warning(f"Could not check/enable RTSP (camera may not support API): {e}")
