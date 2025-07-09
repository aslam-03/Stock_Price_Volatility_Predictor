@echo off
echo ========================================
echo Starting Stock Volatility Predictor
echo Using secure environment variables
echo ========================================

REM Build the Docker image (no credentials in image)
echo Building Docker image...
docker build -t stock-volatility-predictor .

REM Check if build was successful
if %ERRORLEVEL% NEQ 0 (
    echo Build failed! Please check the Dockerfile and try again.
    pause
    exit /b 1
)

echo Build successful! Starting container with secure credentials...

REM Run container with environment file (credentials only at runtime)
docker run -d -p 8501:8501 --env-file .env --name stock-volatility-app stock-volatility-predictor

REM Check if container started successfully
if %ERRORLEVEL% NEQ 0 (
    echo Container failed to start! Checking logs...
    docker logs stock-volatility-app
    pause
    exit /b 1
)

echo ========================================
echo App started successfully!
echo Access at: http://localhost:8501
echo ========================================
echo.
echo Your credentials are secure:
echo - NOT stored in Docker image
echo - Only available during runtime
echo - Safe to share the Docker image
echo.
echo To stop: docker stop stock-volatility-app && docker rm stock-volatility-app
echo ========================================

pause
