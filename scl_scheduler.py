#!/usr/bin/env python3
"""
scl_scheduler.py - Batch Processing Scheduler for SCL

Runs ingestion jobs on a schedule (overnight, hourly, etc.)
Manages job queues, retries, and resource allocation.

Usage:
    # Run overnight batch (starts at 2 AM, stops at 6 AM)
    python scl_scheduler.py --overnight
    
    # Run continuous ingestion with rate limiting
    python scl_scheduler.py --continuous --rate 100
    
    # Schedule specific jobs
    python scl_scheduler.py --schedule jobs.yaml
"""

import os
import sys
import time
import json
import yaml
import signal
import logging
import argparse
import schedule
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SCL-Scheduler")


@dataclass
class ScheduledJob:
    """A job to be scheduled."""
    name: str
    source: str  # wikipedia, arxiv, common_crawl, urls
    config: Dict
    schedule: str  # cron-like or "overnight", "hourly", etc.
    enabled: bool = True
    last_run: float = 0
    run_count: int = 0


class SCLScheduler:
    """
    Scheduler for batch SCL ingestion jobs.
    """
    
    def __init__(
        self,
        db_path: str = "scl_atoms.db",
        api_url: str = "http://localhost:8000",
        state_file: str = "scheduler_state.json"
    ):
        self.db_path = db_path
        self.api_url = api_url
        self.state_file = state_file
        self.jobs: List[ScheduledJob] = []
        self.running = False
        self._stop_event = threading.Event()
        
        # Load state
        self._load_state()
    
    def _load_state(self):
        """Load scheduler state from file."""
        if Path(self.state_file).exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.jobs = [
                        ScheduledJob(**j) for j in state.get("jobs", [])
                    ]
                logger.info(f"Loaded {len(self.jobs)} scheduled jobs")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save scheduler state to file."""
        state = {
            "jobs": [asdict(j) for j in self.jobs],
            "last_save": time.time()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def add_job(self, job: ScheduledJob):
        """Add a job to the schedule."""
        self.jobs.append(job)
        self._save_state()
        logger.info(f"Added job: {job.name}")
    
    def _run_job(self, job: ScheduledJob):
        """Execute a single job."""
        from scl_crawler import SCLCrawler
        
        logger.info(f"Starting job: {job.name}")
        
        crawler = SCLCrawler(
            db_path=self.db_path,
            api_url=self.api_url
        )
        
        try:
            if job.source == "wikipedia":
                if "dump_path" in job.config:
                    crawler.ingest_wikipedia_dump(
                        job.config["dump_path"],
                        limit=job.config.get("limit")
                    )
            
            elif job.source == "arxiv":
                crawler.ingest_arxiv(
                    category=job.config.get("category", "cs"),
                    limit=job.config.get("limit", 1000)
                )
            
            elif job.source == "common_crawl":
                crawler.ingest_common_crawl(
                    num_segments=job.config.get("segments", 5),
                    docs_per_segment=job.config.get("docs_per_segment", 500)
                )
            
            elif job.source == "urls":
                if "urls_file" in job.config:
                    crawler.ingest_url_list(job.config["urls_file"])
            
            job.last_run = time.time()
            job.run_count += 1
            self._save_state()
            
            logger.info(f"Completed job: {job.name}")
            
        except Exception as e:
            logger.error(f"Job {job.name} failed: {e}")
    
    def run_overnight(
        self,
        start_hour: int = 2,
        end_hour: int = 6,
        jobs: List[str] = None
    ):
        """
        Run jobs during overnight window.
        
        Args:
            start_hour: Hour to start (24h format)
            end_hour: Hour to stop
            jobs: List of job names to run (None = all)
        """
        logger.info(f"Overnight mode: {start_hour}:00 - {end_hour}:00")
        
        self.running = True
        
        while self.running and not self._stop_event.is_set():
            now = datetime.now()
            
            # Check if within overnight window
            if start_hour <= now.hour < end_hour:
                # Run enabled jobs
                for job in self.jobs:
                    if not job.enabled:
                        continue
                    if jobs and job.name not in jobs:
                        continue
                    
                    # Check if already run today
                    if job.last_run:
                        last_run_date = datetime.fromtimestamp(job.last_run).date()
                        if last_run_date == now.date():
                            continue
                    
                    self._run_job(job)
                    
                    # Check if still in window
                    if datetime.now().hour >= end_hour:
                        break
            
            else:
                # Wait until start time
                if now.hour >= end_hour:
                    # Wait until tomorrow
                    next_start = now.replace(
                        hour=start_hour, minute=0, second=0
                    ) + timedelta(days=1)
                else:
                    # Wait until start hour today
                    next_start = now.replace(
                        hour=start_hour, minute=0, second=0
                    )
                
                wait_seconds = (next_start - now).total_seconds()
                logger.info(f"Waiting {wait_seconds/3600:.1f} hours until {next_start}")
                
                # Sleep in chunks to allow interruption
                while wait_seconds > 0 and not self._stop_event.is_set():
                    time.sleep(min(60, wait_seconds))
                    wait_seconds -= 60
        
        logger.info("Overnight scheduler stopped")
    
    def run_continuous(
        self,
        rate_per_hour: int = 100,
        jobs: List[str] = None
    ):
        """
        Run jobs continuously with rate limiting.
        
        Args:
            rate_per_hour: Maximum documents per hour
            jobs: List of job names to run
        """
        logger.info(f"Continuous mode: {rate_per_hour} docs/hour")
        
        self.running = True
        docs_this_hour = 0
        hour_start = time.time()
        
        while self.running and not self._stop_event.is_set():
            # Reset counter each hour
            if time.time() - hour_start > 3600:
                docs_this_hour = 0
                hour_start = time.time()
            
            # Check rate limit
            if docs_this_hour >= rate_per_hour:
                sleep_time = 3600 - (time.time() - hour_start)
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping {sleep_time/60:.1f} min")
                    time.sleep(min(60, sleep_time))
                continue
            
            # Run jobs
            for job in self.jobs:
                if not job.enabled:
                    continue
                if jobs and job.name not in jobs:
                    continue
                
                self._run_job(job)
                docs_this_hour += job.config.get("limit", 100)
                
                if docs_this_hour >= rate_per_hour:
                    break
            
            # Small delay between cycles
            time.sleep(10)
        
        logger.info("Continuous scheduler stopped")
    
    def run_scheduled(self):
        """Run jobs based on their individual schedules."""
        logger.info("Starting scheduled mode")
        
        self.running = True
        
        # Set up schedules
        for job in self.jobs:
            if not job.enabled:
                continue
            
            if job.schedule == "hourly":
                schedule.every().hour.do(self._run_job, job)
            elif job.schedule == "daily":
                schedule.every().day.at("02:00").do(self._run_job, job)
            elif job.schedule == "weekly":
                schedule.every().sunday.at("02:00").do(self._run_job, job)
        
        while self.running and not self._stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)
        
        logger.info("Scheduled mode stopped")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        self._stop_event.set()
        logger.info("Scheduler stop requested")
    
    def load_jobs_from_yaml(self, yaml_path: str):
        """Load job definitions from YAML file."""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        for job_config in config.get("jobs", []):
            job = ScheduledJob(
                name=job_config["name"],
                source=job_config["source"],
                config=job_config.get("config", {}),
                schedule=job_config.get("schedule", "daily"),
                enabled=job_config.get("enabled", True)
            )
            self.add_job(job)
        
        logger.info(f"Loaded {len(self.jobs)} jobs from {yaml_path}")


def create_sample_schedule():
    """Create a sample schedule YAML file."""
    sample = {
        "jobs": [
            {
                "name": "arxiv_cs_daily",
                "source": "arxiv",
                "schedule": "daily",
                "enabled": True,
                "config": {
                    "category": "cs",
                    "limit": 500
                }
            },
            {
                "name": "arxiv_physics_weekly",
                "source": "arxiv",
                "schedule": "weekly",
                "enabled": True,
                "config": {
                    "category": "physics",
                    "limit": 1000
                }
            },
            {
                "name": "common_crawl_sample",
                "source": "common_crawl",
                "schedule": "weekly",
                "enabled": False,
                "config": {
                    "segments": 5,
                    "docs_per_segment": 500
                }
            }
        ]
    }
    
    with open("schedule_sample.yaml", 'w') as f:
        yaml.dump(sample, f, default_flow_style=False)
    
    print("Created schedule_sample.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="SCL Batch Processing Scheduler"
    )
    
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Run in overnight mode (2 AM - 6 AM)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously with rate limiting"
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Run based on job schedules"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        help="Path to schedule YAML file"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=100,
        help="Documents per hour for continuous mode (default: 100)"
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=2,
        help="Start hour for overnight mode (default: 2)"
    )
    parser.add_argument(
        "--end-hour",
        type=int,
        default=6,
        help="End hour for overnight mode (default: 6)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="scl_atoms.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="SCL API server URL"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample schedule YAML file"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_schedule()
        return 0
    
    # Initialize scheduler
    scheduler = SCLScheduler(
        db_path=args.db,
        api_url=args.api_url
    )
    
    # Load schedule if provided
    if args.schedule:
        scheduler.load_jobs_from_yaml(args.schedule)
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        scheduler.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run appropriate mode
    if args.overnight:
        scheduler.run_overnight(
            start_hour=args.start_hour,
            end_hour=args.end_hour
        )
    
    elif args.continuous:
        scheduler.run_continuous(rate_per_hour=args.rate)
    
    elif args.scheduled:
        scheduler.run_scheduled()
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
