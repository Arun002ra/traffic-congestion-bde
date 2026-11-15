@echo off
title Deploy to GitHub and Vercel
echo ========================================================
echo   TRAFFIC CONGESTION BIG DATA PROJECT - DEPLOYMENT
echo ========================================================
echo.
set PATH=C:\Users\S ARUNRAJ\AppData\Local\Programs\Git\cmd;%PATH%
cd /d "D:\traffic-congestion-bde"

echo 1. Verifying git repository status...
git add .
git commit -m "Real-Time Traffic Analysis & Congestion Prediction using Big Data (Vercel Ready)" 2>nul

echo.
echo 2. Setting remote origin to https://github.com/Arun002ra/traffic-congestion-bde.git...
git remote remove origin 2>nul
git remote add origin https://github.com/Arun002ra/traffic-congestion-bde.git
git branch -M main

echo.
echo 3. Pushing commits to GitHub...
echo (If a GitHub sign-in window appears, please click 'Sign in with your browser')
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo  SUCCESS! Your repository is now live on GitHub:
    echo  https://github.com/Arun002ra/traffic-congestion-bde
    echo.
    echo  To deploy on Vercel:
    echo  1. Open: https://vercel.com/new
    echo  2. Click 'Import' on traffic-congestion-bde
    echo  3. Click 'Deploy'
    echo ========================================================
) else (
    echo.
    echo --------------------------------------------------------
    echo If push failed, make sure the empty repository exists:
    echo 1. Open https://github.com/new
    echo 2. Name it: traffic-congestion-bde
    echo 3. Click 'Create repository'
    echo 4. Run this script again!
    echo --------------------------------------------------------
)
pause
