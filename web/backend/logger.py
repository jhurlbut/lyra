"""Centralized logging system for Lyra"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

class LyraLogger:
    """Centralized logger that writes to both files and console"""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.loggers = {}
        
        # Create main logger
        self.main_logger = self._setup_logger("lyra_main")
        
    def _setup_logger(self, name: str, level=logging.INFO):
        """Setup a logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(message)s')
        
        # File handler - detailed logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler - simple format for frontend display
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_job_logger(self, job_id: str) -> logging.Logger:
        """Get or create a logger for a specific job"""
        if job_id not in self.loggers:
            # Create job-specific log file
            job_logger = self._setup_logger(f"job_{job_id}")
            self.loggers[job_id] = job_logger
        return self.loggers[job_id]
    
    def log_job(self, job_id: str, message: str, level=logging.INFO, 
                callback: Optional[Callable[[str], None]] = None):
        """Log a message for a specific job"""
        logger = self.get_job_logger(job_id)
        logger.log(level, message)
        
        # Also call the callback if provided (for frontend updates)
        if callback:
            callback(message)
    
    def log_main(self, message: str, level=logging.INFO):
        """Log to the main logger"""
        self.main_logger.log(level, message)
    
    def get_log_file(self, job_id: str) -> Optional[Path]:
        """Get the log file path for a job"""
        for log_file in self.log_dir.glob(f"job_{job_id}_*.log"):
            return log_file
        return None

# Global logger instance
logger = LyraLogger()