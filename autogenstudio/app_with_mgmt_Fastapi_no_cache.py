#!/usr/bin/env python3
"""
Simplified FastAPI API for AutoGen Studio team chat system demo
"""

import os
import json
import logging
import uuid
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# Import team manager and response model
from autogenstudio.teammanager import TeamManager
from autogenstudio.datamodel import Response

# Configure basic logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s %(levelname)s: %(message)s [%(pathname)s:%(lineno)d]'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AutoGen Studio Demo API", version="1.0.0")

# Initialize TeamManager
team_manager = TeamManager()

# Utility function to get team configuration file path
def get_team_config_path():
    """Get the path to the team configuration file"""
    default_path = os.path.join(os.getcwd(), "demo_tooled_config.json")
    team_file_path = os.environ.get("AUTOGENSTUDIO_TEAM_FILE", default_path)
    
    if not os.path.exists(team_file_path):
        logger.warning(f"Team configuration not found at {team_file_path}, using default")
        team_file_path = "demo_tooled_config.json"
    
    logger.info(f"Using team configuration: {team_file_path}")
    return team_file_path

# Simple API key authentication
async def verify_api_key(request: Request):
    """Verify the API key in request headers"""
    # For demo purposes, accept any API key or none at all
    api_key = request.headers.get('X-API-KEY', 'demo-key')
    return True

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests"""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Endpoint: Chat Handler
@app.post("/api/chat")
async def chat_handler(request: Request, authorized: bool = Depends(verify_api_key)):
    """Handle chat requests by passing them to the team manager"""
    try:
        # Parse request data
        data = await request.json()
        
        # Extract required fields with defaults
        user_id = data.get('user_id', str(uuid.uuid4()))
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        message = data.get('message', '').lower()
        
        logger.info(f"Processing message in conversation {conversation_id}: {message[:50]}...")
        
        # Execute team_manager.run with the demo config
        try:
            result = await team_manager.run(
                task=message,
                team_config=get_team_config_path()
            )
        except Exception as e:
            logger.error(f"Team manager error: {e}")
            return JSONResponse(content={
                "conversation_id": conversation_id,
                "response": {
                    "message": "Error processing request",
                    "status": False
                },
                "status": False,
                "timestamp": datetime.now().isoformat()
            })

        # Serialize result
        try:
            serialized_result = {
                "task_result": {
                    "messages": [
                        {
                            "source": getattr(msg, 'source', None),
                            "content": getattr(msg, 'content', str(msg)),
                            "type": getattr(msg, 'type', None)
                        } for msg in getattr(result.task_result, 'messages', [])
                    ],
                    "stop_reason": getattr(result.task_result, 'stop_reason', None)
                },
                "duration": getattr(result, 'duration', None)
            }
        except Exception as e:
            logger.error(f"Result serialization error: {e}")
            serialized_result = {"message": str(result) if result else "Processing completed"}

        # Return response
        response_obj = Response(
            message="Task completed",
            status=True,
            data=serialized_result
        )
        
        return JSONResponse(content=jsonable_encoder({
            "conversation_id": conversation_id,
            "response": response_obj.data,
            "status": response_obj.status,
            "timestamp": datetime.now().isoformat()
        }))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=200,
            content={
                "conversation_id": conversation_id if 'conversation_id' in locals() else str(uuid.uuid4()),
                "response": {
                    "message": "An error occurred while processing your request",
                    "status": False
                },
                "status": False,
                "timestamp": datetime.now().isoformat()
            }
        )

# Endpoint: System Health Check
@app.get("/api/system/health")
async def health_check():
    """Simple health check endpoint"""
    config_path = get_team_config_path()
    config_exists = os.path.exists(config_path)
    
    return JSONResponse(content={
        "status": "operational" if config_exists else "configuration_missing",
        "config_path": config_path,
        "config_exists": config_exists,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)