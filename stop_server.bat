@echo off
echo Stopping Flask server...
curl -X POST http://localhost:5000/shutdown
echo Server stopped!
pause