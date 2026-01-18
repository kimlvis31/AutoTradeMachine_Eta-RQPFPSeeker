#!/bin/bash

# [1]: Check Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual Environment (.venv) Not Found!"
    echo "Please run './setup.sh' first."
    exit 1
fi

# [2]: Activate Virtual Environment
echo "[System] Activating virtual environment..."
source .venv/bin/activate

# [3]: Run main.py
echo "[System] Starting RQPFPSeeker..."
python3 main.py

# [4]: Termination
echo "[System] Program Terminated."