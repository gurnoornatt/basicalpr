#!/usr/bin/env python3
"""
Hotlist Update Script
Downloads and processes stolen/wanted vehicle data into hotlist.csv
"""

import csv
import requests
import argparse
import os
from datetime import datetime

class HotlistUpdater:
    def __init__(self, hotlist_path="hotlist.csv"):
        self.hotlist_path = hotlist_path
    
    def create_sample_hotlist(self):
        """Create a sample hotlist with test data"""
        sample_plates = [
            "ABC123",
            "XYZ789", 
            "TEST001",
            "SAMPLE1",
            "DEMO999"
        ]
        
        with open(self.hotlist_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['plate', 'reason', 'date_added'])  # Header
            
            for plate in sample_plates:
                writer.writerow([plate, 'Sample/Test', datetime.now().strftime('%Y-%m-%d')])
        
        print(f"Created sample hotlist with {len(sample_plates)} plates: {self.hotlist_path}")
    
    def update_from_url(self, url):
        """Download hotlist from URL and update local file"""
        try:
            print(f"Downloading hotlist from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV content
            lines = response.text.strip().split('\n')
            
            with open(self.hotlist_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['plate', 'reason', 'date_updated'])  # Header
                
                for line in lines:
                    if line.strip():
                        # Assume simple format: plate,reason or just plate
                        parts = line.split(',')
                        plate = parts[0].strip().upper()
                        reason = parts[1].strip() if len(parts) > 1 else 'Downloaded'
                        
                        if plate:
                            writer.writerow([plate, reason, datetime.now().strftime('%Y-%m-%d')])
            
            print(f"Updated hotlist from URL: {self.hotlist_path}")
            
        except Exception as e:
            print(f"Error updating from URL: {e}")
            print("Creating sample hotlist instead...")
            self.create_sample_hotlist()
    
    def add_plate(self, plate, reason="Manual addition"):
        """Add a single plate to the hotlist"""
        plate = plate.upper().strip()
        
        # Read existing plates
        existing_plates = set()
        if os.path.exists(self.hotlist_path):
            with open(self.hotlist_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:
                        existing_plates.add(row[0])
        
        # Add new plate if not exists
        if plate not in existing_plates:
            with open(self.hotlist_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([plate, reason, datetime.now().strftime('%Y-%m-%d')])
            print(f"Added plate to hotlist: {plate}")
        else:
            print(f"Plate already in hotlist: {plate}")
    
    def remove_plate(self, plate):
        """Remove a plate from the hotlist"""
        plate = plate.upper().strip()
        
        if not os.path.exists(self.hotlist_path):
            print("Hotlist file not found")
            return
        
        # Read all plates except the one to remove
        updated_rows = []
        removed = False
        
        with open(self.hotlist_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                updated_rows.append(header)
            
            for row in reader:
                if row and row[0] != plate:
                    updated_rows.append(row)
                elif row and row[0] == plate:
                    removed = True
        
        # Write updated file
        with open(self.hotlist_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)
        
        if removed:
            print(f"Removed plate from hotlist: {plate}")
        else:
            print(f"Plate not found in hotlist: {plate}")
    
    def list_plates(self):
        """List all plates in the hotlist"""
        if not os.path.exists(self.hotlist_path):
            print("Hotlist file not found")
            return
        
        with open(self.hotlist_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if header:
                print(f"{'Plate':<12} {'Reason':<20} {'Date':<12}")
                print("-" * 45)
            
            count = 0
            for row in reader:
                if row:
                    plate = row[0] if len(row) > 0 else ""
                    reason = row[1] if len(row) > 1 else ""
                    date = row[2] if len(row) > 2 else ""
                    
                    print(f"{plate:<12} {reason:<20} {date:<12}")
                    count += 1
            
            print(f"\nTotal plates: {count}")

def main():
    parser = argparse.ArgumentParser(description='Hotlist Update Utility')
    parser.add_argument('--hotlist', default='hotlist.csv', help='Hotlist CSV file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create sample hotlist
    subparsers.add_parser('init', help='Create sample hotlist')
    
    # Update from URL
    update_parser = subparsers.add_parser('update', help='Update hotlist from URL')
    update_parser.add_argument('url', help='URL to download hotlist from')
    
    # Add plate
    add_parser = subparsers.add_parser('add', help='Add plate to hotlist')
    add_parser.add_argument('plate', help='License plate to add')
    add_parser.add_argument('--reason', default='Manual addition', help='Reason for adding')
    
    # Remove plate
    remove_parser = subparsers.add_parser('remove', help='Remove plate from hotlist')
    remove_parser.add_argument('plate', help='License plate to remove')
    
    # List plates
    subparsers.add_parser('list', help='List all plates in hotlist')
    
    args = parser.parse_args()
    
    updater = HotlistUpdater(args.hotlist)
    
    if args.command == 'init':
        updater.create_sample_hotlist()
    elif args.command == 'update':
        updater.update_from_url(args.url)
    elif args.command == 'add':
        updater.add_plate(args.plate, args.reason)
    elif args.command == 'remove':
        updater.remove_plate(args.plate)
    elif args.command == 'list':
        updater.list_plates()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 