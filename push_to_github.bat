@echo off
title Push to GitHub - VisionAI
echo ========================================================
echo  Pushing Face Detection project to GitHub...
echo  Repository: https://github.com/adithyanpillai45/face-detection.git
echo ========================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Pushed successfully to GitHub!
) else (
    echo [NOTICE] If prompted for authentication, follow the on-screen instructions or browser prompt.
)
echo.
pause

