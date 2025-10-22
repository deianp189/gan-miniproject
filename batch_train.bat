@echo off
set saveLoc=zdim
@REM for %%l in ("0.001,0.002" "0.002,0.001" "0.002,0.004" "0.004,0.002") do (
@REM   for /f "tokens=1,2 delims=," %%a in (%%l) do (
@REM     python src/train_gan.py --opt adam --lrG %%a --lrD %%b --save-loc %saveLoc%
@REM   )
@REM )
for %%z in (50 150 200) do (
  python src/train_gan.py --opt adam --zdim %%z --save-loc %saveLoc%
)

echo All done!
powershell -Command "[System.Console]::Beep(500, 1000)"