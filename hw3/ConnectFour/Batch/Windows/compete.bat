@echo off&setlocal enabledelayedexpansion
for /r %%i in (.\dll\*) do (
	for /r %%j in (.\TestCases\*) do (
		set v1=%%i
		set v2=%%j
rem		if !v1! LSS !v2! (
rem		echo "compete .\compete_result\!v1:~-14,-4!_!v2:~-6,-4!.txt" 
		echo compete !v1:~-14,-4! !v2:~-6,-4!
		compete !v1! !v2! .\compete_result\!v1:~-14,-4!_!v2:~-6,-4!.txt 1
rem		) else (
rem			echo "!v1! !v2! no" 
rem		)
rem		pause
	)
)
rem pause