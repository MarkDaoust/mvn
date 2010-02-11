#! /bin/bash

python mvar.py > doctest.txt
git diff ./doctest.txt > doctest.diff
gedit doctest.txt

