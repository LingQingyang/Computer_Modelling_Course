Pushing Code Files to GitHub in Windows Batch Script:

@echo off
setlocal enabledelayedexpansion

:: === Base folder for all uploads ===
cd "C:\Users\DELL\Desktop\Code_Files" || (echo [ERROR] Code_Files folder does not exist & pause & exit /b)

:: === Add all changes ===
git add .

:: === Generate commit message with timestamp ===
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set DATE=%%a-%%b-%%c
set MSG=auto commit on %DATE% %time:~0,8%
git commit -m "%MSG%" 2>nul

:: === Ask for GitHub user name ===
set /p USER=Enter GitHub user name (e.g. LingQingyang):

:: === Ask for GitHub repository name ===
set /p REPO=Enter GitHub repository name (e.g. MyProject):

:: === Remote repository address ===
set REMOTE=https://github.com/%USER%/%REPO%.git

:: === Setup remote safely ===
git remote | find "origin" >nul
if %errorlevel% neq 0 (
    git remote add origin %REMOTE%
) else (
    git remote set-url origin %REMOTE%
)

:: === Make sure we are on main branch ===
git checkout main 2>nul || git checkout -b main

:: === Pull the latest changes first ===
echo [INFO] Pulling latest changes from GitHub...
git fetch origin main
git pull origin main --rebase

:: === Push to GitHub ===
echo [INFO] Pushing to remote...
git push origin main

echo.
echo [DONE] All files have been synced safely to: %REMOTE%
pause
