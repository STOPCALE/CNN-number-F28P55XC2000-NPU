################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
_backup_charge_mtl/%.obj: ../_backup_charge_mtl/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'C2000 Compiler - building file: "$<"'
	"E:/ccs/ccs/tools/compiler/ti-cgt-c2000_25.11.1.LTS/bin/cl2000" -v28 -ml -mt --cla_support=cla2 --float_support=fpu32 --tmu_support=tmu1 --vcu_support=vcrc -O2 --fp_mode=relaxed --include_path="E:/desk/CCS/cnn_and_number" --include_path="E:/ccs/C2000Ware_26_01_00_00" --include_path="E:/desk/CCS/cnn_and_number/device" --include_path="E:/ccs/C2000Ware_26_01_00_00/driverlib/f28p55x/driverlib/" --include_path="E:/ccs/ccs/tools/compiler/ti-cgt-c2000_25.11.1.LTS/include" --define=DEBUG --define=_FLASH --diag_suppress=10063 --diag_warning=225 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi --preproc_with_compile --preproc_dependency="_backup_charge_mtl/$(basename $(<F)).d_raw" --include_path="E:/desk/CCS/cnn_and_number/CPU1_FLASH/syscfg" --obj_directory="_backup_charge_mtl" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


