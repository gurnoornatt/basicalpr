#!/usr/bin/env python3
"""
SQLite Database Module for LPR System
Handles all database operations for license plate detection hits
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import time

# Database configuration
DB_PATH = 'hits.db'

class DatabaseManager:
    """Manages all database operations for the LPR system"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self) -> None:
        """
        Creates the database if it doesn't exist with the exact schema from PRD
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create hits table with exact PRD schema
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS hits(
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts     TEXT,          -- ISO 8601 timestamp
                plate  TEXT,          -- OCR text
                image  TEXT,          -- file path to JPEG
                alert  INTEGER        -- 1 if in hotlist
            )''')
            
            # Create index for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate ON hits(plate)')
            
            # Create index on timestamp for efficient purging
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ts ON hits(ts)')
            
            conn.commit()
            conn.close()
            print(f"Database initialized: {self.db_path}")
            
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
            raise
    
    def insert_hit(self, timestamp: str, plate_text: str, image_path: str, alert_status: int) -> bool:
        """
        Inserts detection into database
        
        Args:
            timestamp: ISO 8601 timestamp string
            plate_text: OCR extracted text
            image_path: Path to saved plate image
            alert_status: 1 if plate is in hotlist, 0 otherwise
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            start_time = time.time()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO hits (ts, plate, image, alert) VALUES (?, ?, ?, ?)',
                (timestamp, plate_text, image_path, alert_status)
            )
            
            conn.commit()
            row_id = cursor.lastrowid
            conn.close()
            
            elapsed_time = time.time() - start_time
            
            # Verify meets <1s requirement from PRD
            if elapsed_time >= 1.0:
                print(f"WARNING: Database insertion took {elapsed_time:.3f}s (>1s requirement)")
            
            print(f"Hit inserted successfully (ID: {row_id}, Time: {elapsed_time:.3f}s)")
            return True
            
        except sqlite3.Error as e:
            print(f"Error inserting hit: {e}")
            return False
    
    def get_recent_hits(self, limit: int = 100) -> List[Tuple]:
        """
        Retrieves most recent hits from database
        
        Args:
            limit: Maximum number of hits to retrieve (default: 100)
            
        Returns:
            List of tuples containing hit data (id, ts, plate, image, alert)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, ts, plate, image, alert FROM hits ORDER BY id DESC LIMIT ?',
                (limit,)
            )
            
            hits = cursor.fetchall()
            conn.close()
            
            return hits
            
        except sqlite3.Error as e:
            print(f"Error retrieving recent hits: {e}")
            return []
    
    def get_hits_by_plate(self, plate_text: str) -> List[Tuple]:
        """
        Retrieves all hits for a specific plate number
        
        Args:
            plate_text: Plate number to search for
            
        Returns:
            List of tuples containing hit data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, ts, plate, image, alert FROM hits WHERE plate = ? ORDER BY ts DESC',
                (plate_text,)
            )
            
            hits = cursor.fetchall()
            conn.close()
            
            return hits
            
        except sqlite3.Error as e:
            print(f"Error retrieving hits for plate {plate_text}: {e}")
            return []
    
    def get_alert_hits(self, limit: int = 50) -> List[Tuple]:
        """
        Retrieves recent hits that triggered alerts (hotlist matches)
        
        Args:
            limit: Maximum number of alert hits to retrieve
            
        Returns:
            List of tuples containing alert hit data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, ts, plate, image, alert FROM hits WHERE alert = 1 ORDER BY id DESC LIMIT ?',
                (limit,)
            )
            
            hits = cursor.fetchall()
            conn.close()
            
            return hits
            
        except sqlite3.Error as e:
            print(f"Error retrieving alert hits: {e}")
            return []
    
    def purge_old_hits(self, days: int = 30) -> int:
        """
        Removes hits older than specified days based on retention policy
        
        Args:
            days: Number of days to retain (default: 30)
            
        Returns:
            int: Number of records deleted
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_iso = cutoff_date.isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First count how many will be deleted
            cursor.execute('SELECT COUNT(*) FROM hits WHERE ts < ?', (cutoff_iso,))
            count_to_delete = cursor.fetchone()[0]
            
            if count_to_delete == 0:
                conn.close()
                print(f"No hits older than {days} days found")
                return 0
            
            # Delete old hits
            cursor.execute('DELETE FROM hits WHERE ts < ?', (cutoff_iso,))
            
            conn.commit()
            conn.close()
            
            # Vacuum to reclaim space (must be done outside transaction)
            conn = sqlite3.connect(self.db_path)
            conn.execute('VACUUM')
            conn.close()
            
            print(f"Purged {count_to_delete} hits older than {days} days")
            return count_to_delete
            
        except sqlite3.Error as e:
            print(f"Error purging old hits: {e}")
            return -1
    
    def get_stats(self) -> dict:
        """
        Get database statistics
        
        Returns:
            dict: Database statistics including total hits, alerts, etc.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total hits
            cursor.execute('SELECT COUNT(*) FROM hits')
            total_hits = cursor.fetchone()[0]
            
            # Total alerts
            cursor.execute('SELECT COUNT(*) FROM hits WHERE alert = 1')
            total_alerts = cursor.fetchone()[0]
            
            # Date range
            cursor.execute('SELECT MIN(ts), MAX(ts) FROM hits')
            date_range = cursor.fetchone()
            
            # Most recent hit
            cursor.execute('SELECT ts, plate FROM hits ORDER BY id DESC LIMIT 1')
            recent_hit = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_hits': total_hits,
                'total_alerts': total_alerts,
                'date_range': {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                },
                'most_recent': {
                    'timestamp': recent_hit[0] if recent_hit else None,
                    'plate': recent_hit[1] if recent_hit else None
                }
            }
            
        except sqlite3.Error as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def close(self):
        """Close database connection (for cleanup)"""
        # This implementation uses connection-per-operation pattern
        # so no persistent connection to close
        pass


# Convenience functions for backward compatibility and ease of use
def init_db():
    """Initialize database (convenience function)"""
    db = DatabaseManager()
    return db

def insert_hit(timestamp: str, plate_text: str, image_path: str, alert_status: int) -> bool:
    """Insert hit (convenience function)"""
    db = DatabaseManager()
    return db.insert_hit(timestamp, plate_text, image_path, alert_status)

def get_recent_hits(limit: int = 100) -> List[Tuple]:
    """Get recent hits (convenience function)"""
    db = DatabaseManager()
    return db.get_recent_hits(limit)

def purge_old_hits(days: int = 30) -> int:
    """Purge old hits (convenience function)"""
    db = DatabaseManager()
    return db.purge_old_hits(days)


if __name__ == "__main__":
    # Test the database module
    print("Testing Database Module...")
    
    db = DatabaseManager()
    
    # Test insertion
    test_timestamp = datetime.now().isoformat()
    success = db.insert_hit(test_timestamp, "TEST123", "plates/test.jpg", 1)
    print(f"Insertion test: {'PASSED' if success else 'FAILED'}")
    
    # Test retrieval
    recent = db.get_recent_hits(5)
    print(f"Retrieval test: {'PASSED' if len(recent) > 0 else 'FAILED'}")
    print(f"Retrieved {len(recent)} recent hits")
    
    # Test stats
    stats = db.get_stats()
    print(f"Stats test: {'PASSED' if stats else 'FAILED'}")
    print(f"Database stats: {stats}")
    
    print("Database module test complete!") 