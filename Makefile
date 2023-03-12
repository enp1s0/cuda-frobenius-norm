NVCC=nvcc
NVCCFLAGS=-std=c++17
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-I./include/

TARGET=cunorm.test

$(TARGET):test/main.cu src/cunorm.cu
	$(NVCC) $+ -o $@ $(NVCCFLAGS)
  
clean:
	rm -f $(TARGET)
