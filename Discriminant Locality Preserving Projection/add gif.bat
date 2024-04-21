@echo off
setlocal enabledelayedexpansion

rem 遍历当前文件夹中含有 "subject" 的文件
for %%f in (*subject*) do (
    rem 获取文件名和扩展名
    for /f "tokens=1,* delims=." %%a in ("%%f") do (
        set "filename=%%a"
        set "extension=%%b"
    )

    rem 重命名文件，在扩展名前添加 .gif
    ren "%%f" "!filename!.!extension!.gif"
)

echo Renaming complete.
pause



