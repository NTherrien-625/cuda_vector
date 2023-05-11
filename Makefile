vector_kernel.o: vector_kernel.cu
	nvcc -c vector_kernel.cu

vector.o: vector.cu
	nvcc -c vector.cu

driver.o: driver.cu
	nvcc -c driver.cu

driver: vector_kernel.o vector.o driver.o
	nvcc vector_kernel.o vector.o driver.o -o driver

clean:
	rm -f *.o
	rm -f driver
