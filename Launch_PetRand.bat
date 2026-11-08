@echo off
REM ============================================================================
REM Launch_PetRand.bat (Windows Double-Click Launcher)
REM Double-click this file in File Explorer to launch the unified PetRand
REM Console, Everyday Randomness Studio, and the Zoomies! game
REM ============================================================================

cd /d "%~dp0"
set PORT=8080

echo ================================================================
echo  Starting PetRand // Everyday Randomness Studio ^& Zoomies!
echo ================================================================
echo   - Pet Arena ^& Everyday Studio : http://127.0.0.1:%PORT%/
echo   - Zoomies! game: http://127.0.0.1:%PORT%/zoomies
echo ================================================================

start "" "http://127.0.0.1:%PORT%/"
python -m petrand.cli serve --host 127.0.0.1 --port %PORT% --sensor sim
pause
