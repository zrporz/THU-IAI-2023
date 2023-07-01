setlocal enabledelayedexpansion
call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\Auxiliary\\Build\\vcvars64.bat"
for /d %%i in (.\Sourcecode\*) do (
cl %%i\*.cpp /GS /GL /W3 /Gy /Zc:wchar_t /Zi /Gm- /O2 /Fd".\\TempResult\\vc142.pdb" /Zc:inline /fp:precise /D "_WINDLL" /D "_UNICODE" /D "UNICODE" /errorReport:prompt /WX- /Zc:forScope /Gd /Oi /MD /FC /Fa".\\TempResult\\" /EHsc /nologo /Fo".\\TempResult\\" /c /TP /Fp".\\TempResult\\Strategy.pch" /diagnostics:column
link .\TempResult\*.obj /OUT:".\\TempResult\\Strategy.dll" /DLL /MANIFEST /LTCG:incremental /NXCOMPAT /PDB:".\\TempResult\\Strategy.pdb" /DYNAMICBASE /DEBUG /MACHINE:X64 /OPT:REF /SUBSYSTEM:CONSOLE /MANIFESTUAC:"level='asInvoker' uiAccess='false'" /ManifestFile:".\\TempResult\\Strategy.dll.intermediate.manifest" /OPT:ICF /ERRORREPORT:PROMPT /NOLOGO /TLBID:1
copy .\TempResult\Strategy.dll .\dll\
mt /outputresource:".\dll\Strategy.dll;#2" /manifest .\TempResult\Strategy.dll.intermediate.manifest /nologo
set a=%%i
ren .\dll\Strategy.dll !a:.\Sourcecode\=!.dll
del /q .\TempResult\*.*
)
