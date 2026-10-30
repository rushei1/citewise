#!/bin/sh
# Start the API in the background, then Streamlit in the foreground on the Space's port.
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
exec streamlit run frontend/app.py \
    --server.address=0.0.0.0 --server.port=7860 \
    --server.enableXsrfProtection=false --server.enableCORS=false \
    --browser.gatherUsageStats=false
