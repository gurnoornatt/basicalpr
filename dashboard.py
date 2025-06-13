#!/usr/bin/env python3
"""
FastAPI Dashboard Backend for License Plate Recognition System
Serves API endpoints for viewing detection history and images
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import uvicorn
from datetime import datetime

app = FastAPI(title="LPR Dashboard", description="License Plate Recognition Dashboard API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve plate images
if os.path.exists("plates"):
    app.mount("/plates", StaticFiles(directory="plates"), name="plates")

@app.get("/")
async def get_dashboard():
    """Serve the main dashboard page"""
    dashboard_path = "static/index.html"
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    else:
        return {"message": "Dashboard not found. Please ensure static/index.html exists."}

@app.get("/hits")
async def get_hits(limit: int = 100):
    """Get recent license plate detections"""
    try:
        if not os.path.exists('hits.db'):
            return {"error": "Database not found. Please run the LPR system first."}
            
        conn = sqlite3.connect('hits.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, ts, plate, image, alert 
            FROM hits 
            ORDER BY ts DESC 
            LIMIT ?
        """, (limit,))
        
        hits = []
        for row in cursor.fetchall():
            hit = dict(row)
            # Format timestamp for display
            try:
                ts = datetime.fromisoformat(hit['ts'])
                hit['formatted_time'] = ts.strftime("%Y-%m-%d %H:%M:%S")
            except:
                hit['formatted_time'] = hit['ts']
            hits.append(hit)
            
        conn.close()
        return {"hits": hits, "count": len(hits)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/img/{filename}")
async def get_image(filename: str):
    """Serve plate images"""
    # Security: only allow jpg files and prevent directory traversal
    if not filename.endswith('.jpg') or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    image_path = os.path.join("plates", filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(
        image_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
    )

@app.get("/stats")
async def get_stats():
    """Get detection statistics"""
    try:
        if not os.path.exists('hits.db'):
            return {"total_detections": 0, "alerts": 0}
            
        conn = sqlite3.connect('hits.db')
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute("SELECT COUNT(*) as total FROM hits")
        total = cursor.fetchone()[0]
        
        # Alert count (though we're not using alerts)
        cursor.execute("SELECT COUNT(*) as alerts FROM hits WHERE alert = 1")
        alerts = cursor.fetchone()[0]
        
        # Recent detections (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) as recent 
            FROM hits 
            WHERE datetime(ts) > datetime('now', '-1 day')
        """)
        recent = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_detections": total,
            "alerts": alerts,
            "recent_24h": recent
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_exists": os.path.exists('hits.db'),
        "plates_dir_exists": os.path.exists('plates')
    }

if __name__ == "__main__":
    print("Starting LPR Dashboard...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    ) 