"""
agents/scheduler.py

Background scheduler that periodically calls agent_manager.refresh_all().
Runs every 30 minutes. Each agent defines its own refresh_hours interval.

Start by calling start_scheduler(agent_manager) once at app startup.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

_scheduler_thread: threading.Thread = None
_stop_event = threading.Event()

CHECK_INTERVAL_SECONDS = 30 * 60  # check every 30 min


def start_scheduler(agent_manager) -> None:
    """
    Start the background scheduler thread.
    Safe to call multiple times — only starts one thread.
    """
    global _scheduler_thread

    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        logger.info("[Scheduler] Already running.")
        return

    _stop_event.clear()

    def _loop():
        logger.info("[Scheduler] Started. Will check every 30 minutes.")
        while not _stop_event.is_set():
            try:
                agent_manager.refresh_all()
            except Exception as e:
                logger.warning(f"[Scheduler] refresh_all error: {e}")
            # Sleep in small increments so stop_event responds quickly
            for _ in range(CHECK_INTERVAL_SECONDS):
                if _stop_event.is_set():
                    break
                time.sleep(1)
        logger.info("[Scheduler] Stopped.")

    _scheduler_thread = threading.Thread(target=_loop, daemon=True, name="agent-scheduler")
    _scheduler_thread.start()


def stop_scheduler() -> None:
    _stop_event.set()
