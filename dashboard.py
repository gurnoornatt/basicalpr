#!/usr/bin/env python3
"""
LPR Dashboard - Web interface for viewing license plate detections
FastAPI-based dashboard as specified in PRD
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any
import argparse

app = FastAPI(title="LPR Dashboard", description="License Plate Recognition Dashboard")

# Global configuration
DB_PATH = "hits.db"
PLATES_DIR = "plates"

def get_db_connection():
    """Get database connection"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found")
    return sqlite3.connect(DB_PATH)

@app.get("/hits", response_model=List[Dict[str, Any]])
async def get_hits(limit: int = 100):
    """Get last N hits from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, ts, plate, image, alert
            FROM hits
            ORDER BY ts DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        hits = []
        for row in rows:
            hits.append({
                "id": row[0],
                "timestamp": row[1],
                "plate": row[2],
                "image": row[3],
                "alert": bool(row[4])
            })
        
        return hits
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/img/{filename}")
async def get_image(filename: str):
    """Serve plate images"""
    image_path = os.path.join(PLATES_DIR, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total hits
        cursor.execute("SELECT COUNT(*) FROM hits")
        total_hits = cursor.fetchone()[0]
        
        # Alert hits
        cursor.execute("SELECT COUNT(*) FROM hits WHERE alert = 1")
        alert_hits = cursor.fetchone()[0]
        
        # Today's hits
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) FROM hits WHERE ts LIKE ?", (f"{today}%",))
        today_hits = cursor.fetchone()[0]
        
        # Recent unique plates
        cursor.execute("SELECT COUNT(DISTINCT plate) FROM hits WHERE ts >= datetime('now', '-24 hours')")
        unique_plates_24h = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_hits": total_hits,
            "alert_hits": alert_hits,
            "today_hits": today_hits,
            "unique_plates_24h": unique_plates_24h
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LPR Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                text-align: center;
            }
            .stats-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }
            .stat-label {
                color: #7f8c8d;
                margin-top: 5px;
            }
            .hits-container {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .hits-header {
                background-color: #34495e;
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .hits-table {
                width: 100%;
                border-collapse: collapse;
            }
            .hits-table th,
            .hits-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
            }
            .hits-table th {
                background-color: #ecf0f1;
                font-weight: 600;
            }
            .hits-table tr:hover {
                background-color: #f8f9fa;
            }
            .alert-badge {
                background-color: #e74c3c;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
            }
            .plate-text {
                font-family: monospace;
                font-weight: bold;
                font-size: 1.1em;
            }
            .timestamp {
                color: #7f8c8d;
                font-size: 0.9em;
            }
            .image-thumb {
                width: 60px;
                height: 40px;
                object-fit: cover;
                border-radius: 4px;
                cursor: pointer;
            }
            .refresh-btn {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
            }
            .refresh-btn:hover {
                background-color: #2980b9;
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚗 License Plate Recognition Dashboard</h1>
            <p>Real-time monitoring of detected license plates</p>
        </div>

        <div class="stats-container" id="statsContainer">
            <div class="stat-card">
                <div class="stat-number" id="totalHits">-</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="alertHits">-</div>
                <div class="stat-label">Alert Hits</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="todayHits">-</div>
                <div class="stat-label">Today's Hits</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="uniquePlates">-</div>
                <div class="stat-label">Unique Plates (24h)</div>
            </div>
        </div>

        <div class="hits-container">
            <div class="hits-header">
                <h3>Recent Detections</h3>
                <button class="refresh-btn" onclick="loadData()">Refresh</button>
            </div>
            <div id="hitsContent">
                <div class="loading">Loading...</div>
            </div>
        </div>

        <script>
            let refreshInterval;

            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    document.getElementById('totalHits').textContent = stats.total_hits;
                    document.getElementById('alertHits').textContent = stats.alert_hits;
                    document.getElementById('todayHits').textContent = stats.today_hits;
                    document.getElementById('uniquePlates').textContent = stats.unique_plates_24h;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }

            async function loadHits() {
                try {
                    const response = await fetch('/hits');
                    const hits = await response.json();
                    
                    let tableHTML = `
                        <table class="hits-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Plate</th>
                                    <th>Image</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;
                    
                    hits.forEach(hit => {
                        const timestamp = new Date(hit.timestamp).toLocaleString();
                        const imageName = hit.image.split('/').pop() || '';
                        const alertBadge = hit.alert ? '<span class="alert-badge">ALERT</span>' : '';
                        
                        tableHTML += `
                            <tr>
                                <td><div class="timestamp">${timestamp}</div></td>
                                <td><div class="plate-text">${hit.plate}</div></td>
                                <td>
                                    ${imageName ? `<img src="/img/${imageName}" class="image-thumb" alt="Plate image" onclick="window.open('/img/${imageName}', '_blank')">` : 'No image'}
                                </td>
                                <td>${alertBadge}</td>
                            </tr>
                        `;
                    });
                    
                    tableHTML += '</tbody></table>';
                    
                    if (hits.length === 0) {
                        tableHTML = '<div class="loading">No detections found</div>';
                    }
                    
                    document.getElementById('hitsContent').innerHTML = tableHTML;
                } catch (error) {
                    console.error('Error loading hits:', error);
                    document.getElementById('hitsContent').innerHTML = '<div class="loading">Error loading data</div>';
                }
            }

            async function loadData() {
                await Promise.all([loadStats(), loadHits()]);
            }

            // Auto-refresh every 5 seconds
            function startAutoRefresh() {
                refreshInterval = setInterval(loadData, 5000);
            }

            function stopAutoRefresh() {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
            }

            // Load data on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadData();
                startAutoRefresh();
            });

            // Stop refresh when page is hidden
            document.addEventListener('visibilitychange', function() {
                if (document.hidden) {
                    stopAutoRefresh();
                } else {
                    startAutoRefresh();
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

def main():
    parser = argparse.ArgumentParser(description='LPR Dashboard Server')
    parser.add_argument('--host', default='localhost', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--db', default='hits.db', help='Database file path')
    parser.add_argument('--plates-dir', default='plates', help='Plates directory')
    
    args = parser.parse_args()
    
    # Update global configuration
    global DB_PATH, PLATES_DIR
    DB_PATH = args.db
    PLATES_DIR = args.plates_dir
    
    # Run the server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 