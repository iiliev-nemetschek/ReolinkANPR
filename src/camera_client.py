"""Reolink camera client with ANPR capabilities."""

import asyncio
import math
import re
import time
from collections import deque
from contextlib import aclosing
import cv2
import numpy as np
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple
from reolink_aio.api import Host
from .logger import logger

# RTSP frame streaming for ANPR
STREAM_FPS = 5  # Frames per second kept from the camera's ~25; ALPR can't keep up with more
PRE_TRIGGER_SECONDS = 2.0  # Frames from this long before a detection are still scanned
STREAM_CONNECT_TIMEOUT = 8  # Seconds a scan waits for its first frame when nothing is buffered
STREAM_STALL_TIMEOUT = 15  # Reconnect if the stream delivers no data for this long
KEEPALIVE_SECONDS = 5  # The first camera API call after ~20s of silence takes 1-2s; poll to keep it fast


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

        # Persistent RTSP stream (see start_stream)
        self._stream_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._frames = deque(maxlen=math.ceil(STREAM_FPS * PRE_TRIGGER_SECONDS) + 2)  # (monotonic time, JPEG)
        self._frame_seq = 0
        self._frame_event = asyncio.Event()
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
        await self.stop_stream()

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

    def start_stream(self):
        """Keep one RTSP connection open and buffer recent frames (persistent stream option).

        The camera needs several seconds to hand out a first frame on a fresh connection,
        so a detection must never wait for one: it scans the buffer and live frames instead.
        Also starts a light API poll so the AI-state check at detection time isn't a cold call.
        Does nothing when the persistent stream is turned off in the config.
        """
        if not self.config.persistent_stream:
            return
        self._start_ffmpeg()
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def _start_ffmpeg(self):
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_supervisor())

    @staticmethod
    async def _cancel_task(task: asyncio.Task, timeout: float = 10):
        """Cancel a background task, but never let a stuck one block the caller (e.g. a config reload)."""
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("A background task did not stop in time; continuing")

    async def _stop_ffmpeg(self):
        task, self._stream_task = self._stream_task, None
        if task:
            await self._cancel_task(task)
        self._frames.clear()

    async def stop_stream(self):
        task, self._keepalive_task = self._keepalive_task, None
        if task:
            await self._cancel_task(task)
        await self._stop_ffmpeg()

    async def _keepalive_loop(self):
        while True:
            await asyncio.sleep(KEEPALIVE_SECONDS)
            try:
                await self.host.get_ai_state(self.config.camera_channel)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Keepalive failed: {e}")

    def _ffmpeg_stream_cmd(self) -> List[str]:
        # Reolink uses 1-based channel numbering
        rtsp_url = f"rtsp://{self.config.camera_username}:{self.config.camera_password}@{self.config.camera_host}:554/h264Preview_{self.config.camera_channel+1:02d}_main"
        return [
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

    async def _stream_supervisor(self):
        """Run the ffmpeg stream forever, reconnecting with backoff when it drops."""
        backoff = 1
        while True:
            started = time.monotonic()
            try:
                error = await self._run_stream()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                error = self._mask_credentials(str(e))

            logger.warning(f"RTSP stream ended ({error or 'no error output'}); reconnecting in {backoff}s")
            if time.monotonic() - started > 60:
                backoff = 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _run_stream(self) -> str:
        """Read one ffmpeg session into the frame buffer. Returns ffmpeg's error output."""
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *self._ffmpeg_stream_cmd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stderr_tail = bytearray()  # Only the end of ffmpeg's error output is kept

        async def drain_stderr():
            while True:
                data = await process.stderr.read(4096)
                if not data:
                    return
                stderr_tail.extend(data)
                del stderr_tail[:-4000]

        async def discard_stdout():
            while await process.stdout.read(262144):
                pass

        stderr_reader = asyncio.create_task(drain_stderr())
        buffer = bytearray()
        connected = False
        stalled = False

        try:
            while True:
                chunk = await asyncio.wait_for(process.stdout.read(262144), timeout=STREAM_STALL_TIMEOUT)
                if not chunk:
                    break
                buffer += chunk
                # Each JPEG starts with FFD8 and ends with FFD9
                while True:
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9', start + 2) if start >= 0 else -1
                    if start < 0 or end < 0:
                        break
                    self._frames.append((time.monotonic(), bytes(buffer[start:end + 2])))
                    del buffer[:end + 2]
                    self._frame_seq += 1
                    self._frame_event.set()
                    if not connected:
                        connected = True
                        logger.info(f"RTSP stream connected - first frame after {time.monotonic() - started:.2f}s")
        except asyncio.TimeoutError:
            stalled = True
        finally:
            if process.returncode is None:
                process.kill()
            # asyncio only reports the process as exited once both pipes hit EOF, and it stops
            # reading a pipe nobody drains - so keep draining here or wait() can hang forever.
            try:
                await asyncio.wait_for(
                    asyncio.gather(discard_stdout(), stderr_reader, process.wait(), return_exceptions=True),
                    timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("ffmpeg did not exit cleanly after being stopped")

        if stalled:
            return f"no data for {STREAM_STALL_TIMEOUT}s"
        return self._mask_credentials(stderr_tail.decode(errors='replace').strip())

    async def stream_frames(self, duration_seconds: float) -> AsyncIterator[bytes]:
        """Yield frames for one scan.

        With the persistent stream on, recent buffered frames come first, then live ones.
        With it off, ffmpeg is started for this scan only and stopped when the scan ends.
        """
        on_demand = not self.config.persistent_stream
        if on_demand:
            self._start_ffmpeg()
        else:
            self.start_stream()  # No-op when the stream is already running

        try:
            async with aclosing(self._scan_frames(duration_seconds)) as frames:
                async for frame in frames:
                    yield frame
        finally:
            if on_demand:
                await self._stop_ffmpeg()

    async def _scan_frames(self, duration_seconds: float) -> AsyncIterator[bytes]:
        """Buffered frames newest-first, then live ones.

        Live frames that arrive while the consumer is busy are skipped, so processing never
        falls behind. The scan lasts duration_seconds; if nothing is buffered it starts counting
        at the first live frame (waiting up to STREAM_CONNECT_TIMEOUT for it).
        """
        started = time.monotonic()
        buffered = [frame for taken, frame in reversed(self._frames) if started - taken <= PRE_TRIGGER_SECONDS]
        last_seq = self._frame_seq
        got_frame = bool(buffered)
        deadline = started + (duration_seconds if buffered else STREAM_CONNECT_TIMEOUT)
        logger.info(f"Scanning with {len(buffered)} buffered frames")

        for frame in buffered:
            if time.monotonic() >= deadline:
                return
            yield frame

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not got_frame:
                    logger.error(f"No frame from RTSP stream within {STREAM_CONNECT_TIMEOUT}s")
                return

            if self._frame_seq == last_seq:
                self._frame_event.clear()
                try:
                    await asyncio.wait_for(self._frame_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
                continue

            last_seq = self._frame_seq
            if not got_frame:
                got_frame = True
                deadline = time.monotonic() + duration_seconds
                logger.info(f"First live frame after {time.monotonic() - started:.2f}s")
            yield self._frames[-1][1]

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
