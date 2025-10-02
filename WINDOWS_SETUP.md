# Running on Windows

This guide provides specific instructions for setting up and running the CFG Visualizer on Windows.

## Prerequisites

### 1. Install Python
- Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
- During installation, make sure to check "Add Python to PATH"
- Verify installation:
  ```cmd
  python --version
  ```

### 2. Install Node.js
- Download Node.js 14 or higher from [nodejs.org](https://nodejs.org/)
- The installer includes npm automatically
- Verify installation:
  ```cmd
  node --version
  npm --version
  ```

## Setup Instructions

### Backend Setup

1. Open Command Prompt or PowerShell
2. Navigate to the backend directory:
   ```cmd
   cd backend
   ```

3. Install Python dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```cmd
   python app.py
   ```

The backend will start on `http://localhost:5000`

### Frontend Setup

1. Open a **new** Command Prompt or PowerShell window
2. Navigate to the frontend directory:
   ```cmd
   cd frontend
   ```

3. Install Node.js dependencies:
   ```cmd
   npm install
   ```
   This may take a few minutes.

4. Start the development server:
   ```cmd
   npm start
   ```

The frontend will automatically open in your browser at `http://localhost:3000`

## Common Windows Issues

### Issue: 'python' is not recognized
**Solution**: Make sure Python is added to PATH during installation, or use `py` instead of `python`:
```cmd
py app.py
```

### Issue: 'npm' is not recognized
**Solution**: Restart your Command Prompt after installing Node.js, or add Node.js to PATH manually.

### Issue: Port already in use
**Solution**: 
- For backend (port 5000): Stop other applications using port 5000
- For frontend (port 3000): The system will offer to use a different port

### Issue: Permission denied errors
**Solution**: Run Command Prompt as Administrator (right-click > Run as Administrator)

## Building for Production

To create a production build of the frontend:

```cmd
cd frontend
npm run build
```

The build files will be in the `frontend/build` directory. You can serve them using:

```cmd
npm install -g serve
serve -s build
```

## Firewall Settings

If you can't access the application:
1. Windows Defender Firewall might be blocking the ports
2. Go to: Windows Defender Firewall > Advanced Settings > Inbound Rules
3. Create new rules to allow:
   - Port 5000 (Backend)
   - Port 3000 (Frontend)

## Using PowerShell

All commands work in PowerShell as well. If you prefer PowerShell:

```powershell
# Backend
cd backend
pip install -r requirements.txt
python app.py

# Frontend (in a new PowerShell window)
cd frontend
npm install
npm start
```

## Tips for Windows Users

1. **Use Git Bash**: If you have Git installed, you can use Git Bash which provides a Unix-like terminal experience
2. **VS Code Terminal**: If using VS Code, the integrated terminal works great for running these commands
3. **Keep terminals open**: Keep both backend and frontend terminals open while using the application
