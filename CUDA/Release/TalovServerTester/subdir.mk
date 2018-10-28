################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../TalovServerTester/main.cu 

OBJS += \
./TalovServerTester/main.o 

CU_DEPS += \
./TalovServerTester/main.d 


# Each subdirectory must supply rules for building sources it contributes
TalovServerTester/%.o: ../TalovServerTester/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30  -odir "TalovServerTester" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


