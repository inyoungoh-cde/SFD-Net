@echo off
rem ---------------------------------------------------------------------------
rem Builds LGD and produces the ready-to-run Windows package:
rem
rem     deploy\LGD_portable\      LGD.exe + VC++ runtime DLLs + README + data
rem     deploy\LGD_portable.zip   the same folder, zipped for a GitHub release
rem
rem Ordinary builds (Visual Studio, plain msbuild) do NOT create these - the
rem packaging step is opt-in via /p:MakeDeployPackage=true, which this script
rem passes for you.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "MSBUILD="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
	for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe 2^>nul`) do set "MSBUILD=%%i"
)
if not defined MSBUILD (
	set "MSBUILD=%ProgramFiles(x86)%\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
)
if not exist "%MSBUILD%" (
	echo ERROR: MSBuild was not found. Open a Developer Command Prompt and run:
	echo        msbuild LGD.sln /p:Configuration=Release /p:Platform=x64 /p:MakeDeployPackage=true
	pause
	exit /b 1
)

echo Using MSBuild: %MSBUILD%
echo.
"%MSBUILD%" LGD.sln /p:Configuration=Release /p:Platform=x64 /p:MakeDeployPackage=true /v:minimal /nologo
if errorlevel 1 (
	echo.
	echo BUILD FAILED - package not created.
	pause
	exit /b 1
)

echo.
echo Zipping deploy\LGD_portable ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path 'deploy\LGD_portable' -DestinationPath 'deploy\LGD_portable.zip' -Force"
if errorlevel 1 (
	echo WARNING: could not create the zip file.
) else (
	echo Created: deploy\LGD_portable.zip
)

echo.
echo Done.
pause
