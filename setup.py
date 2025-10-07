#!/usr/bin/env python3
"""
Neural Showcase Setup Script
Installs all dependencies and prepares the project for running
"""

import subprocess
import sys
import os

def run_command(command, cwd=None):
    """Run a command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, check=True, cwd=cwd, 
                              capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🧠 Neural Showcase Setup")
    print("=" * 50)
    
    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install Python dependencies")
        sys.exit(1)
    
    # Install frontend dependencies
    print("\n📦 Installing frontend dependencies...")
    frontend_path = os.path.join("web", "frontend")
    if not run_command("npm install", cwd=frontend_path):
        print("Failed to install frontend dependencies")
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\nTo run the Neural Showcase:")
    print("1. Start backend: python backend.py")
    print("2. Start frontend: cd web/frontend && npm start")
    print("3. Open http://localhost:3000 in your browser")

if __name__ == "__main__":
    main()