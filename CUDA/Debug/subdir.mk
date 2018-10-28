################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

OBJS += \
./main.o 

CU_DEPS += \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -I/usr/local/include -I/usr/local/include/c++/7.1.0 -I/usr/include/c++/4.2.1 -I/usr/include -I/usr/local/lib/gcc/x86_64-apple-darwin15.6.0/7.1.0/include -I/Developer/NVIDIA/CUDA-9.2/include -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -I/usr/local/include -I/usr/local/include/c++/7.1.0 -I/usr/include/c++/4.2.1 -I/usr/include -I/usr/local/lib/gcc/x86_64-apple-darwin15.6.0/7.1.0/include -I/Developer/NVIDIA/CUDA-9.2/include -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


