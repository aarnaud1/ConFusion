#!/usr/bin/bash

find . -regex '.*\.\(cpp\|hpp\|cc\|cxx\|cuh\|cu\|inl\)' -exec clang-format -style=file -i {} \;