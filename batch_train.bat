@echo off
set saveLoc=default
for %%o in (adabelief lion) do (
  @REM python src/train_gan.py --opt sgd --momentum %%b --save-loc %saveLoc%
  python src/train_gan.py --opt %%o --save-loc %saveLoc%
)

echo All done!
powershell -Command "[System.Console]::Beep(500, 1000)"