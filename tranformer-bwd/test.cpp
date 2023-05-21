#include "order_control.h"


template<typename T>
void load_file(T* data, const char* filename, int size) {
	float* buffer = new float[size];
	FILE* f = fopen(filename, "rb");
	fread((void*)buffer, sizeof(float), size, f);
	fclose(f);
	for (int i = 0; i < size; i++) {
		data[i] = T(buffer[i]);
	}
	delete[]buffer;
}

int main() {
	int tt_ranks[7] = { 1, 16, 30, 30, 30, 16, 1 };
	int tt_shapes[6] = { 16,8,8,8,8,16 };

	TYPE_WEIGHT tt_cores[num_cores * WD];
	TYPE_WEIGHT grad_cores[num_cores * WD];
	TYPE_DATA input[16 * 8 * 8 * Batchsize];
	TYPE_DATA grad_output[8 * 8 * 16 * Batchsize];

	load_file(input, "input.bin",16 * 8 * 8 * Batchsize);
	load_file(grad_output, "grad_output.bin", 16 * 8 * 8 * Batchsize);

	load_file(tt_cores, "weight0.bin", 1 * 16 * 16);
	load_file(tt_cores + WD, "weight1.bin", 16 * 8 * 30);
	load_file(tt_cores + WD * 2, "weight2.bin", 30 * 8 * 30);
	load_file(tt_cores + WD * 3, "weight3.bin", 30 * 8 * 30);
	load_file(tt_cores + WD * 4, "weight4.bin", 30 * 8 * 16);
	load_file(tt_cores+ WD * 5, "weight5.bin", 16 * 16 * 1);

	order_control_tt_grad(tt_cores, tt_ranks, tt_shapes, input, grad_output, grad_cores);


}