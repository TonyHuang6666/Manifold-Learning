@echo off
setlocal enabledelayedexpansion

for /d %%d in (*) do (
    set "counter=1"
    pushd "%%d"
    for %%f in (*.gif) do (
        ren "%%~nf%%~xf" "!counter!.gif"
        set /a "counter+=1"
    )
    popd
)

echo All .gif files in subfolders have been renamed successfully.


