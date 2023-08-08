# Copyright (C) 2023 Adrien ARNAUD
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

STD := c++17
OPT_LVL := -O3

CXX := g++ -g
NVCC := nvcc -std=$(STD) -ccbin=$(CXX)

HOST_CFLAGS := -std=$(STD) -Wall -Wextra -ffast-math -fopenmp $(OPT_LVL)
HOST_LFLAGS := -Wl\,-rpath\,./GL/lib -lglfw -lgflags \
								-L./GL/lib -lOpenGL \
								-L./lib -lgpuFusion \
								-lopencv_imgcodecs -lopencv_core -lopencv_imgproc \
								-lopencv_highgui -ltinyply
NVCC_FLAGS := $(OPT_LVL) --machine=64 --use_fast_math \
							--default-stream=per-thread \
							--expt-relaxed-constexpr --expt-extended-lambda \
							-arch=native -rdc=true \
							-gencode=arch=compute_75,code=sm_75 \
							-Xcompiler "$(HOST_CFLAGS)" \
							-Xcudafe --diag_suppress=declared_but_not_referenced
IFLAGS := -I./include \
					-I./main/include \
					-I./GL/include \
					-I/usr/include/opencv4 \
					-I/usr/local/cuda/include
LFLAGS := -Xcompiler "$(HOST_LFLAGS)" \
					-lcuda -lcudart -lcudadevrt -lstdc++ 

CU_OBJ_FILES :=  $(patsubst src/%.cu,obj/%.cu.o,$(wildcard src/**/*.cu))
OBJ_FILES := $(patsubst src/%.cpp,obj/%.o,$(wildcard src/**/*.cpp))
MAIN_SRC_FILES := main/DataStreamer.cpp main/DepthMapDataset.cpp

MODULE := lib/libgpuFusion.so
EXEC := bin/main bin/test_edges bin/test_harris

## -----------------------------------------------------------------------------

all: deps $(EXEC)

deps:
	$(shell mkdir -p bin/)
	$(shell mkdir -p obj/)
	$(shell mkdir -p obj/fusion)
	$(shell mkdir -p obj/imgproc)
	$(shell mkdir -p obj/io)
	$(shell mkdir -p lib/)

obj/%.cu.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) -Xcompiler '-fPIC' $(IFLAGS) -dc $< -o $@

obj/%.o: src/%.cpp
	$(CXX) $(HOST_CFLAGS) -c -fPIC $(IFLAGS) -o $@ $<

$(MODULE): $(CU_OBJ_FILES) $(OBJ_FILES)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^

bin/main: main/main.cpp $(MODULE)
	$(CXX) $(HOST_CFLAGS) $(IFLAGS) -o $@ $^ $(MAIN_SRC_FILES) \
	$(HOST_LFLAGS) -L/usr/local/cuda/lib64 -lcuda -lcudart

bin/test_%: main/test_%.cpp $(MODULE)
	$(CXX) $(HOST_CFLAGS) $(IFLAGS) -o $@ $^ $(MAIN_SRC_FILES) \
	$(HOST_LFLAGS) -L/usr/local/cuda/lib64 -lcuda -lcudart

clean:
	rm -rfd bin
	rm -rfd obj
	rm -rfd lib
