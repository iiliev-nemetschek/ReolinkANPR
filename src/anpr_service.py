"""Main ANPR service orchestrating camera and processing."""

import asyncio
import signal
import time
from contextlib import aclosing
from pathlib import Path
from .logger import logger
from .config import Config
from .camera_client import CameraClient
from .alpr_processor import ALPRProcessor
from .database import Database
from .notifier import Notifier


class ANPRService:
    """Main ANPR service coordinating all components."""

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = Config(config_path)
        self.config.ensure_directories()

        # Setup logger with config
        from .logger import setup_logger
        global logger
        logger = setup_logger(
            "ReolinkANPR",
            self.config.log_file,
            self.config.log_level
        )

        # Initialize components
        self.camera = CameraClient(self.config)
        self.alpr = ALPRProcessor(self.config)
        self.database = Database(self.config.database_path)
        self.notifier = Notifier(self.config)

        self.running = False
        self.last_detection_time = 0
        self.detection_cooldown = 30  # seconds - ignore new detections for 30s after starting a recording
        self._background_tasks = set()  # keeps fire-and-forget tasks (settings restore) alive

    async def start(self):
        """Start the ANPR service."""
        logger.info("=" * 60)
        logger.info("ReolinkANPR - Starting Service")
        logger.info("=" * 60)

        try:
            # Initialize database
            await self.database.initialize()
            logger.info("Database initialized")

            # Check if config is still default
            if self.config.camera_password == "CHANGE_ME":
                logger.warning("=" * 60)
                logger.warning("Configuration needed!")
                logger.warning("Please configure your camera at:")
                logger.warning("  http://localhost:5001/config")
                logger.warning("Then restart ReolinkANPR")
                logger.warning("=" * 60)
                # Keep running but wait for config
                self.running = True
                await self._wait_for_config()
                return

            # Connect to camera with retry logic
            connected = False
            retry_count = 0
            max_retries = 3

            while not connected and retry_count < max_retries:
                if retry_count > 0:
                    logger.info(f"Retry {retry_count}/{max_retries} - Attempting to connect...")
                    await asyncio.sleep(5)

                connected = await self.camera.connect()
                retry_count += 1

            if not connected:
                logger.error("=" * 60)
                logger.error("Failed to connect to camera after 3 attempts")
                logger.error("Please check:")
                logger.error("  - Camera IP address is correct")
                logger.error("  - Camera is powered on and connected")
                logger.error("  - Username and password are correct")
                logger.error("  - RTSP is enabled on camera")
                logger.error("")
                logger.error("Configure at: http://localhost:5001/config")
                logger.error("=" * 60)
                # Keep running but wait
                self.running = True
                await self._wait_for_config()
                return

            # Set up event-driven processing callback
            self.camera.set_processing_callback(self._handle_vehicle_detection)

            # Start monitoring
            self.running = True
            logger.info("ANPR Service started successfully")
            logger.info(f"Monitoring {self.config.camera_name} for vehicles...")
            logger.info("=" * 60)
            
            # Start reload signal checker
            asyncio.create_task(self._check_reload_signal())

            await self._monitoring_loop()

        except Exception as e:
            logger.error(f"Failed to start ANPR service: {e}")
            raise

    async def reload(self):
        """Reload configuration and reinitialize components without stopping."""
        global logger
        
        logger.info("=" * 60)
        logger.info("Reloading ANPR Service Configuration...")
        logger.info("=" * 60)
        
        try:
            # Reload config file
            old_config_path = self.config.config_path
            self.config = Config(str(old_config_path))
            logger.info("✓ Configuration file reloaded")
            
            # Reconnect camera with new settings
            if self.camera:
                await self.camera.disconnect()
                logger.info("✓ Disconnected from camera")
            
            self.camera = CameraClient(self.config)
            connected = await self.camera.connect()
            
            if connected:
                logger.info("✓ Reconnected to camera with new settings")
                # Re-set the processing callback
                self.camera.set_processing_callback(self._handle_vehicle_detection)
            else:
                logger.error("✗ Failed to reconnect to camera")
                return False
            
            # Reload ALPR processor with new models
            self.alpr = ALPRProcessor(self.config)
            logger.info("✓ ALPR processor reloaded")
            
            # Reload notifier with new settings
            self.notifier = Notifier(self.config)
            logger.info("✓ Notifier reloaded")
            
            # Update logger level if changed
            from .logger import setup_logger
            logger = setup_logger(
                "ReolinkANPR",
                self.config.log_file,
                self.config.log_level
            )
            logger.info("✓ Logger updated")
            
            logger.info("=" * 60)
            logger.info("Configuration Reloaded Successfully!")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    async def stop(self):
        """Stop the ANPR service gracefully."""
        logger.info("Stopping ANPR Service...")
        self.running = False

        # Disconnect from camera
        await self.camera.disconnect()

        logger.info("ANPR Service stopped")

    async def _wait_for_config(self):
        """Wait indefinitely for user to configure via web interface."""
        logger.info("Service running in configuration mode...")
        logger.info("Waiting for configuration via web interface")

        # Saving the config from the web UI drops a reload signal; apply it here
        await self._check_reload_signal()

    async def _check_reload_signal(self):
        """Check for reload signal file periodically."""
        reload_signal_path = Path(".") / ".reload_signal"
        
        while self.running:
            try:
                if reload_signal_path.exists():
                    logger.info("Reload signal detected!")
                    reload_signal_path.unlink()  # Remove signal file
                    await self.reload()
            except Exception as e:
                logger.error(f"Error checking reload signal: {e}")
            
            await asyncio.sleep(2)  # Check every 2 seconds

    async def _monitoring_loop(self):
        """Main monitoring loop - wait for TCP push events."""
        logger.info("Monitoring with TCP push events (Baichuan protocol)")
        logger.info("Waiting for vehicle detection events...")

        # Keep alive loop - TCP push events handled by callback
        while self.running:
            try:
                # Just keep the loop alive - events come via TCP push callback
                await asyncio.sleep(60)  # Wake up every minute to check if still running

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _handle_vehicle_detection(self):
        """Handle TCP push event - stream frames from the camera and read the plate."""
        before_task = None
        scanning = False
        try:
            # Check cooldown - prevent multiple recordings for same vehicle
            current_time = time.time()
            time_since_last = current_time - self.last_detection_time
            
            if time_since_last < self.detection_cooldown:
                logger.info(f"🚫 Detection cooldown active ({time_since_last:.1f}s < {self.detection_cooldown}s) - skipping")
                return
            
            # Update last detection time
            self.last_detection_time = current_time
            scanning = True
            started = time.monotonic()

            logger.info("Vehicle detected - scanning RTSP stream")

            # Camera settings change runs alongside the stream; it never blocks plate reading
            if getattr(self.config, 'before_recording_enabled', False):
                before_task = asyncio.create_task(self.camera.apply_recording_settings('before'))

            result = await self._scan_stream()
            logger.info(f"⏱ Scan finished {time.monotonic() - started:.2f}s after detection")

            if result:
                await self._publish_detection(result)
                logger.info(f"⏱ Event published {time.monotonic() - started:.2f}s after detection")
            else:
                logger.info("No valid plates found")

        except Exception as e:
            logger.error(f"Error handling vehicle detection: {e}")
        finally:
            if scanning:
                self._schedule_settings_restore(before_task)

    async def _scan_stream(self):
        """Feed live frames to ALPR until a plate is read consistently or the scan window ends."""
        scan = self.alpr.new_scan()
        save_dir = Path(self.config.database_path).parent

        async with aclosing(self.camera.stream_frames(self.config.recording_duration)) as frames:
            async for frame in frames:
                if await asyncio.to_thread(scan.add_frame, frame):
                    logger.info(f"Stopping early at frame {scan.frames_seen}: plate read consistently")
                    break

        logger.info(f"Analyzed {scan.frames_seen} frames")
        return await asyncio.to_thread(scan.finish, save_dir)

    def _schedule_settings_restore(self, before_task):
        """Restore camera settings in the background, after the 'before' change has landed."""
        if not getattr(self.config, 'after_recording_enabled', False):
            return

        async def restore():
            if before_task is not None:
                if await before_task:
                    logger.info("✅ Before-recording settings applied")
                else:
                    logger.warning("⚠️ Before-recording settings failed")

            logger.info("Restoring after-recording settings...")
            if await self.camera.apply_recording_settings('after'):
                logger.info("✅ After-recording settings restored")
            else:
                logger.warning("⚠️ After-recording settings failed")

        task = asyncio.create_task(restore())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _publish_detection(self, result: dict):
        """Save a recognized plate and send notifications."""
        try:
            save_dir = Path(self.config.database_path).parent

            logger.info(
                f"Plate recognized: {result['plate_number']} "
                f"(confidence: {result['confidence']:.2%})"
            )

            # Save to database
            event_id = await self.database.add_event(result)
            logger.info(f"Event saved to database (ID: {event_id})")

            # Send notifications (only if it was actually saved, not a duplicate)
            if event_id:
                # Convert relative image path to absolute for notifications
                image_path = result.get('image_path')
                if image_path:
                    image_path = str(save_dir / image_path)

                await self.notifier.send_detection(
                    result['plate_number'],
                    result['confidence'],
                    image_path
                )

        except Exception as e:
            logger.error(f"Error processing detection: {e}")


async def main():
    """Main entry point for the service."""
    service = ANPRService()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(service.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
