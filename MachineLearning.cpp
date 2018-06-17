#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define INPUTS 21 //Between 1 and 50
//#define INPUTS 2 //Between 1 and 50
#define HIDDEN_LAYERS 3 //Between 0 and 3
#define NEURONS_LAYER_1 4 //Between 1 and 30 (Hidden) (Third)
#define NEURONS_LAYER_2 8 //Between 1 and 30 (Hidden) (Second)
#define NEURONS_LAYER_3 15 //Between 1 and 30 (Hidden) (First)
#define NEURONS_LAYER_4 11 //Between 1 and 30 (Outputs)
//#define NEURONS_LAYER_4 2 //Between 1 and 30 (Outputs)

float fatv(float x) {
    return 1/(1+exp(-x));
}

float dfatv(float x) {
    return (fatv(x)*(1-fatv(x)));
}

void print_disease(float output[NEURONS_LAYER_4]) {
	int index = 0;
	float result = 0;
	for (int i = 0; i < NEURONS_LAYER_4; i++) {
		//cout << output[i] << " ";
		if (output[i] >= result) {
			result = output[i];
			index = i;
		}
	}
	//cout << endl;
	if (index == 0)
		cout << "Asthma" << endl;
	if (index == 1)
		cout << "Acute Bronchitis" << endl;
	if (index == 2)
		cout << "Chronic Bronchitis" << endl;
	if (index == 3)
		cout << "Acute Sinusitis" << endl;
	if (index == 4)
		cout << "Chronic Sinusitis" << endl;
	if (index == 5)
		cout << "Rhinitis" << endl;
	if (index == 6)
		cout << "Pneumonia" << endl;
	if (index == 7)
		cout << "Tuberculosis" << endl;
	if (index == 8)
		cout << "Emphysema" << endl;
	if (index == 9)
		cout << "Whooping Cough" << endl;
	if (index == 10)
		cout << "Common Cold" << endl;	
}

int main() {
    //Variables for the neural network
    float input[INPUTS] = { };
    
    float w1[30][51] = { }; //[Neuron][Input]
    float z1[30] = { }; //[Neuron]
    float y1[30] = { }; //[Neuron]
    float w2[30][31] = { };
    float z2[30] = { };
    float y2[30] = { };
    float w3[30][31] = { };
    float z3[30] = { };
    float y3[30] = { };
    float w4[30][51] = { };
    float z4[30] = { };
    float y4[30] = { };
    
    float s[30] = { };
    
    float learning_rate = 0.01;
    
    float reference[30] = { };
    float error[30] = { };
    float g4[30] = { };
    float g3[30] = { };
    float g2[30] = { };
    float g1[30] = { };
    
    //Other variables needed
    float stop_condition = 0.999; //Between 0 and 1. How much does the neural network have to be right on the training so that we can say it "learned"?
    float batch_grade = 0.0;  //The evaluation on the neural network's weights after a complete batch. Training stops if this surpasses the stop condition.
    float mse = 0.0;
    int num_iterations = 0;
    int max_iterations = 1000000;
    bool show_progress_data = false;
    int update_step = 100000;
    bool determination = false;
    float delta_w1 = 0.0;
    float delta_w2 = 0.0;
    float delta_w3 = 0.0;
    float delta_w4 = 0.0;
    float weight_stagnation_threshold = 0.0;
    float previous_batch_grade = 0.0;
    float grade_stagnation_threshold = 0.0;
    int num_restarts = 0;
    
    //Setting Training Data (Batch)
    //First dimension: number of different samples;
    //Second dimension: inputs per sample plus outputs.
    //On XOR example, a batch is a 4x3 array => [0 0 0; 0 1 1; 1 0 1; 1 1 0]
    //int sample_size = 4;
    int sample_size = 49;
    //float batch[sample_size][INPUTS + NEURONS_LAYER_4] = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0}; //XOR
    //float batch[sample_size][INPUTS + NEURONS_LAYER_4] = {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1}; //AND
    //float batch[sample_size][INPUTS + NEURONS_LAYER_4] = {0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}; //XOR+AND
    //float batch[sample_size][INPUTS + NEURONS_LAYER_4] = {0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1}; //x2
    //float batch[sample_size][INPUTS + NEURONS_LAYER_4] = {0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}; //True
    float batch[sample_size][INPUTS + NEURONS_LAYER_4] = {1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0,1,		//Asthma #1
														  1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0,0,		//Asthma #3
														  1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0,0,		//Asthma #4
														  1,0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0,0,		//Acute Bronchitis #1
														  1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0,0,		//Acute Bronchitis #2
														  1,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0,0,		//Acute Bronchitis #3
														  1,0,1,1,1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0,		//Chronic Bronchitis #1
														  1,0,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0,		//Chronic Bronchitis #2
														  1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0,		//Chronic Bronchitis #4
														  1,1,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #1
														  1,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #2
														  1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #4
														  0,1,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #5
														  0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #6
														  0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #7
														  0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #8
														  1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #1
														  1,1,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #2
														  1,0,0,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #3
														  1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #4
														  0,1,0,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #5
														  0,0,0,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #7
														  0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #8
														  0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0,		//Rhinitis #1
														  0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0,		//Rhinitis #2
                                                          0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0,		//Rhinitis #4
														  1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0, 0,0,0,0,0,0,1,0,0,0,0,		//Pneumonia #1
                                                          1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0, 0,0,0,0,0,0,1,0,0,0,0,		//Pneumonia #2
                                                          1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0, 0,0,0,0,0,0,1,0,0,0,0,		//Pneumonia #3
                                                          1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,1,0,0,0,0,		//Pneumonia #4
                                                          1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0,1,0,0,0,0,		//Pneumonia #5
                                                          1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,1, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #1
                                                          1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,1, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #2
                                                          1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,1, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #3
                                                          1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,1, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #4
                                                          1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #5
                                                          1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #6
                                                          1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #7
														  1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,1,0,0,		//Emphysema #1
                                                          1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,1,0,0,		//Emphysema #3
                                                          1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,0,0,		//Emphysema #4
                                                          1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,1,0,		//Whooping Cough #2
                                                          1,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,1,0,		//Whooping Cough #3
                                                          1,0,0,1,1,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,1,0,		//Whooping Cough #4
                                                          1,0,0,1,1,0,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,1,		//Flu #2
                                                          1,0,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,1,		//Flu #3
                                                          1,0,0,0,1,0,1,0,1,1,0,1,0,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,1,		//Flu #4
                                                          1,0,0,0,1,0,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,1,		//Flu #5
														  1,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,1};	//Flu #6

	if (show_progress_data) {
		cout << "Training data:" << endl;
		for (int i = 0; i < sample_size; i++) {
			for (int j = 0; j < (INPUTS + NEURONS_LAYER_4); j++) {
				cout << batch[i][j] << " ";
			}
			cout << endl;
		}
    }
    
    //Setting Test Data (test_data)
    //Same amount of inputs and outputs as batch, any amount of samples.
    //E.g. test_data[?][INPUTS + NEURONS_LAYER_4] = { ... };
    //Don't test the neural network with the same data used to train it.
    int test_size = 11;
    //int test_size = sample_size;
    //float test_data[test_size][INPUTS + NEURONS_LAYER_4] = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0};
    float test_data[test_size][INPUTS + NEURONS_LAYER_4] = {1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0,0,		//Asthma #2
														    1,0,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0,0,		//Acute Bronchitis #4
														    1,0,1,1,0,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0,		//Chronic Bronchitis #3
														    1,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,0,0,		//Acute Sinusitis #3
														    0,1,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,0,0,		//Chronic Sinusitis #6
														    0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0,		//Rhinitis #3
                                                            1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,0,0,0,0,		//Pneumonia #6
                                                            1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,1,0,0,0,		//Tuberculosis #8
                                                            1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,0,0,		//Emphysema #2
														    1,0,0,1,1,0,1,0,1,0,0,1,1,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,1,0,		//Whooping Cough #1
															1,0,0,1,1,0,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,1};		//Flu #1
	if (show_progress_data) {
		cout << "Test data:" << endl;
		for (int i = 0; i < test_size; i++) {
			for (int j = 0; j < (INPUTS + NEURONS_LAYER_4); j++) {
				cout << test_data[i][j] << " ";
			}
			cout << endl;
		}
    }
    
    //Intializing weights
	srand(time(NULL));
	
    while (true) {
		//cout << "W1.1.1 = " << w1[1][1] << endl;
		for (int i = 0; i < 30; i++) {
			for (int j = 0; j < 51; j++) {
				w1[i][j] = ((float) rand() / (RAND_MAX));
			}
		}
		//cout << "W1.1.1 = " << w1[1][1] << endl;
		for (int i = 0; i < 30; i++) {
			for (int j = 0; j < 31; j++) {
				w2[i][j] = ((float) rand() / (RAND_MAX));
				w3[i][j] = ((float) rand() / (RAND_MAX));
			}
		}
		for (int i = 0; i < 30; i++) {
			for (int j = 0; j < 51; j++) {
				w4[i][j] = ((float) rand() / (RAND_MAX));
			}
		}
		
		/*w3[0][0] = 0.7;
		w3[0][1] = 0.3;
		w3[0][2] = 0.8;
		w3[1][0] = 0.2;
		w3[1][1] = 0.5;
		w3[1][2] = 0.6;
		
		w4[0][0] = 0.9;
		w4[0][1] = 0.1;
		w4[0][2] = 0.4;
		w4[1][0] = 0.6;
		w4[1][1] = 0.5;
		w4[1][2] = 0.3;*/
		
		//Training
		while ((batch_grade <= stop_condition) & (num_iterations < max_iterations)) {
			num_iterations++;
			if ((show_progress_data) | (num_iterations % update_step == 0)) {
				cout << "Epoch number " << num_iterations << "! START!" << endl;
			}
			previous_batch_grade = batch_grade;
			batch_grade = 0.0;
			//Single batch
			for (int i = 0; i < sample_size; i++) {
				//Updating input array
				for (int j = 0; j < INPUTS; j++) {
					input[j] = batch[i][j];
				}
				//Updating reference array (for the future)
				for (int j = 0; j < NEURONS_LAYER_4; j++) {
					reference[j] = batch[i][INPUTS+j];
				}
				
				if ((show_progress_data) | (num_iterations % update_step == 0)) {
					//Showing inputs and expected outputs:
					cout << "Inputs:" << endl;
					for (int j = 0; j < INPUTS; j++) {
						cout << input[j] << " ";
					}
					cout << endl;
					cout << "Expected outputs:" << endl;
					//Updating reference array (for the future)
					for (int j = 0; j < NEURONS_LAYER_4; j++) {
						cout << reference[j] << " ";
					}
					cout << endl;
				}
				
				if (HIDDEN_LAYERS == 0) {
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
						z4[j] = -w4[j][0];
						for (int k = 0; k < INPUTS; k++) {
							z4[j] = z4[j] + w4[j][k+1]*input[k];
						}
						y4[j] = fatv(z4[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
						s[j] = y4[j];
						error[j] = reference[j]-s[j];
						batch_grade = batch_grade + pow(error[j],2);
						g4[j] = dfatv(z4[j])*error[j];
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Backpropagation
						if (fabs(learning_rate*g4[j]*(-1)) > delta_w4) {
							delta_w4 = fabs(learning_rate*g4[j]*(-1));
						}
						w4[j][0] = w4[j][0] + learning_rate*g4[j]*(-1);
						for (int k = 0; k < INPUTS; k++) {
							if (fabs(learning_rate*g1[j]*input[k]) > delta_w4) {
								delta_w4 = fabs(learning_rate*g4[j]*input[k]);
							}
							w4[j][k+1] = w4[j][k+1] + learning_rate*g4[j]*input[k];
						}
					}
				} else if (HIDDEN_LAYERS == 1) {
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
						z3[j] = -w3[j][0];
						for (int k = 0; k < INPUTS; k++) {
							z3[j] = z3[j] + w3[j][k+1]*input[k];
						}
						//cout << "z3[" << j << "] = " << z3[j] << endl;
						y3[j] = fatv(z3[j]);
						//cout << "y3[" << j << "] = " << y3[j] << endl;
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
						z4[j] = -w4[j][0];
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							z4[j] = z4[j] + w4[j][k+1]*y3[k];
						}
						//cout << "z4[" << j << "] = " << z4[j] << endl;
						y4[j] = fatv(z4[j]);
						//cout << "y4[" << j << "] = " << y4[j] << endl;
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
						s[j] = y4[j];
						//cout << "s[" << j << "] = " << s[j] << endl;
						error[j] = reference[j]-s[j];
						//cout << "error[" << j << "] = " << error[j] << endl;
						batch_grade = batch_grade + pow(error[j],2);
						g4[j] = dfatv(z4[j])*error[j];
						//cout << "g4[" << j << "] = " << g4[j] << endl;
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Gradient calculation
						g3[j] = 0;
						for (int k = 0; k < NEURONS_LAYER_4; k++) {
							//cout << "Multiplying: " << g4[k] << " * " << w4[k][j+1] << "..." << endl;
							g3[j] = g3[j] + dfatv(z3[j])*(g4[k]*w4[k][j+1]);
						}
						//cout << "g3[" << j << "] = " << g3[j] << endl;
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Backpropagation
						if (fabs(learning_rate*g4[j]*(-1)) > delta_w4) {
							delta_w4 = fabs(learning_rate*g4[j]*(-1));
						}
						w4[j][0] = w4[j][0] + learning_rate*g4[j]*(-1);
						//cout << "w4[" << j << "][0] = " << w4[j][0] << endl;
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							if (fabs(learning_rate*g4[j]*y3[k]) > delta_w4) {
								delta_w4 = fabs(learning_rate*g4[j]*y3[k]);
							}
							w4[j][k+1] = w4[j][k+1] + learning_rate*g4[j]*y3[k];
							//cout << "w4[" << j << "][" <<  k+1 << "] = " << w4[j][k+1] << endl;
						}
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Backpropagation
						if (fabs(learning_rate*g3[j]*(-1)) > delta_w3) {
							delta_w3 = fabs(learning_rate*g3[j]*(-1));
						}
						w3[j][0] = w3[j][0] + learning_rate*g3[j]*(-1);
						//cout << "w3[" << j << "][0] = " << w3[j][0] << endl;
						for (int k = 0; k < INPUTS; k++) {
							if (fabs(learning_rate*g3[j]*input[k]) > delta_w3) {
								delta_w3 = fabs(learning_rate*g3[j]*input[k]);
							}
							w3[j][k+1] = w3[j][k+1] + learning_rate*g3[j]*input[k];
							//cout << "w3[" << j << "][" <<  k+1 << "] = " << w3[j][k+1] << endl;
						}
					}
				} else if (HIDDEN_LAYERS == 2) {
					for (int j = 0; j < NEURONS_LAYER_2; j++) { //Frontpropagation
						z2[j] = -w2[j][0];
						for (int k = 0; k < INPUTS; k++) {
							z2[j] = z2[j] + w2[j][k+1]*input[k];
						}
						y2[j] = fatv(z2[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
						z3[j] = -w3[j][0];
						for (int k = 0; k < NEURONS_LAYER_2; k++) {
							z3[j] = z3[j] + w3[j][k+1]*y2[k];
						}
						y3[j] = fatv(z3[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
						z4[j] = -w4[j][0];
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							z4[j] = z4[j] + w4[j][k+1]*y3[k];
						}
						y4[j] = fatv(z4[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
						s[j] = y4[j];
						error[j] = reference[j]-s[j];
						batch_grade = batch_grade + pow(error[j],2);
						g4[j] = dfatv(z4[j])*error[j];
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Gradient calculation
						g3[j] = 0;
						for (int k = 0; k < NEURONS_LAYER_4; k++) {
							g3[j] = g3[j] + dfatv(z3[j])*(g4[k]*w4[k][j+1]);
						}
					}
					for (int j = 0; j < NEURONS_LAYER_2; j++) { //Gradient calculation
						g2[j] = 0;
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							g2[j] = g2[j] + dfatv(z2[j])*(g3[k]*w3[k][j+1]);
						}
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Backpropagation
						if (fabs(learning_rate*g4[j]*(-1)) > delta_w4) {
							delta_w4 = fabs(learning_rate*g4[j]*(-1));
						}
						w4[j][0] = w4[j][0] + learning_rate*g4[j]*(-1);
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							if (fabs(learning_rate*g4[j]*y3[k]) > delta_w4) {
								delta_w4 = fabs(learning_rate*g4[j]*y3[k]);
							}
							w4[j][k+1] = w4[j][k+1] + learning_rate*g4[j]*y3[k];
						}
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Backpropagation
						if (fabs(learning_rate*g3[j]*(-1)) > delta_w3) {
							delta_w3 = fabs(learning_rate*g3[j]*(-1));
						}
						w3[j][0] = w3[j][0] + learning_rate*g3[j]*(-1);
						for (int k = 0; k < NEURONS_LAYER_2; k++) {
							if (fabs(learning_rate*g3[j]*y1[k]) > delta_w3) {
								delta_w3 = fabs(learning_rate*g3[j]*y2[k]);
							}
							w3[j][k+1] = w3[j][k+1] + learning_rate*g3[j]*y2[k];
						}
					}
					for (int j = 0; j < NEURONS_LAYER_2; j++) { //Backpropagation
						if (fabs(learning_rate*g2[j]*(-1)) > delta_w2) {
							delta_w2 = fabs(learning_rate*g2[j]*(-1));
						}
						w2[j][0] = w2[j][0] + learning_rate*g2[j]*(-1);
						for (int k = 0; k < INPUTS; k++) {
							if (fabs(learning_rate*g2[j]*input[k]) > delta_w2) {
								delta_w2 = fabs(learning_rate*g2[j]*input[k]);
							}
							w2[j][k+1] = w2[j][k+1] + learning_rate*g2[j]*input[k];
						}
					}
				} else if (HIDDEN_LAYERS == 3) {
					for (int j = 0; j < NEURONS_LAYER_1; j++) { //Frontpropagation
						z1[j] = -w1[j][0];
						for (int k = 0; k < INPUTS; k++) {
							z1[j] = z1[j] + w1[j][k+1]*input[k];
						}
						y1[j] = fatv(z1[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_2; j++) { //Frontpropagation
						z2[j] = -w2[j][0];
						for (int k = 0; k < NEURONS_LAYER_1; k++) {
							z2[j] = z2[j] + w2[j][k+1]*y1[k];
						}
						y2[j] = fatv(z2[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
						z3[j] = -w3[j][0];
						for (int k = 0; k < NEURONS_LAYER_2; k++) {
							z3[j] = z3[j] + w3[j][k+1]*y2[k];
						}
						y3[j] = fatv(z3[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
						z4[j] = -w4[j][0];
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							z4[j] = z4[j] + w4[j][k+1]*y3[k];
						}
						y4[j] = fatv(z4[j]);
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
						s[j] = y4[j];
						error[j] = reference[j]-s[j];
						batch_grade = batch_grade + pow(error[j],2);
						g4[j] = dfatv(z4[j])*error[j];
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Gradient calculation
						g3[j] = 0;
						for (int k = 0; k < NEURONS_LAYER_4; k++) {
							g3[j] = g3[j] + dfatv(z3[j])*(g4[k]*w4[k][j+1]);
						}
					}
					for (int j = 0; j < NEURONS_LAYER_2; j++) { //Gradient calculation
						g2[j] = 0;
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							g2[j] = g2[j] + dfatv(z2[j])*(g3[k]*w3[k][j+1]);
						}
					}
					for (int j = 0; j < NEURONS_LAYER_1; j++) { //Gradient calculation
						g1[j] = 0;
						for (int k = 0; k < NEURONS_LAYER_2; k++) {
							g1[j] = g1[j] + dfatv(z1[j])*(g2[k]*w2[k][j+1]);
						}
					}
					for (int j = 0; j < NEURONS_LAYER_4; j++) { //Backpropagation
						if (fabs(learning_rate*g4[j]*(-1)) > delta_w4) {
							delta_w4 = fabs(learning_rate*g4[j]*(-1));
						}
						w4[j][0] = w4[j][0] + learning_rate*g4[j]*(-1);
						for (int k = 0; k < NEURONS_LAYER_3; k++) {
							if (fabs(learning_rate*g4[j]*y3[k]) > delta_w4) {
								delta_w4 = fabs(learning_rate*g4[j]*y3[k]);
							}
							w4[j][k+1] = w4[j][k+1] + learning_rate*g4[j]*y3[k];
						}
					}
					for (int j = 0; j < NEURONS_LAYER_3; j++) { //Backpropagation
						if (fabs(learning_rate*g3[j]*(-1)) > delta_w3) {
							delta_w3 = fabs(learning_rate*g3[j]*(-1));
						}
						w3[j][0] = w3[j][0] + learning_rate*g3[j]*(-1);
						for (int k = 0; k < NEURONS_LAYER_2; k++) {
							if (fabs(learning_rate*g3[j]*y2[k]) > delta_w3) {
								delta_w3 = fabs(learning_rate*g3[j]*y2[k]);
							}
							w3[j][k+1] = w3[j][k+1] + learning_rate*g3[j]*y2[k];
						}
					}
					for (int j = 0; j < NEURONS_LAYER_2; j++) { //Backpropagation
						if (fabs(learning_rate*g2[j]*(-1)) > delta_w2) {
							delta_w2 = fabs(learning_rate*g2[j]*(-1));
						}
						w2[j][0] = w2[j][0] + learning_rate*g2[j]*(-1);
						for (int k = 0; k < NEURONS_LAYER_1; k++) {
							if (fabs(learning_rate*g2[j]*y1[k]) > delta_w2) {
								delta_w2 = fabs(learning_rate*g2[j]*y1[k]);
							}
							w2[j][k+1] = w2[j][k+1] + learning_rate*g2[j]*y1[k];
						}
					}
					for (int j = 0; j < NEURONS_LAYER_1; j++) { //Backpropagation
						if (fabs(learning_rate*g1[j]*(-1)) > delta_w1) {
							delta_w1 = fabs(learning_rate*g1[j]*(-1));
						}
						w1[j][0] = w1[j][0] + learning_rate*g1[j]*(-1);
						for (int k = 0; k < INPUTS; k++) {
							if (fabs(learning_rate*g1[j]*input[k]) > delta_w1) {
								delta_w1 = fabs(learning_rate*g1[j]*input[k]);
							}
							w1[j][k+1] = w1[j][k+1] + learning_rate*g1[j]*input[k];
						}
					}
				}
				
				if ((show_progress_data) | (num_iterations % update_step == 0)) {
					cout << "Outputs:" << endl;
					for (int j = 0; j < NEURONS_LAYER_4; j++) {
						cout << s[j] << " ";
					}
					cout << endl;
				}
			} //End batch
			
			mse = (batch_grade/(sample_size*NEURONS_LAYER_4));
			batch_grade = 1 - mse;
			if ((show_progress_data) | (num_iterations % update_step == 0) | (batch_grade >= stop_condition) | (num_iterations == max_iterations)) {
				cout << "MSE:" << mse << endl;
				cout << "Batch grade: " << batch_grade << endl;
			}
			
			//cout << "Delta-W: " << delta_w1 << " " << delta_w2 << " " << delta_w3 << " " << delta_w4 << endl;
			//out << "Batch Grade: " << batch_grade << endl;
			//cout << "Previous Batch Grade: " << previous_batch_grade << endl;
			//cout << "Delta-Grade: " << fabs(previous_batch_grade-batch_grade) << endl;
			if((delta_w1 < weight_stagnation_threshold) & (delta_w2 < weight_stagnation_threshold) & (delta_w3 < weight_stagnation_threshold) & (delta_w4 < weight_stagnation_threshold)) {
				cout << "Network stagnated due to weights!" << endl;
				break;
			} else if (fabs(previous_batch_grade-batch_grade) < grade_stagnation_threshold) {
				cout << "Network stagnated due to grade!" << endl;
				break;
			} else {
				delta_w1 = 0;
				delta_w2 = 0;
				delta_w3 = 0;
				delta_w4 = 0;
			}
		}
		
		if (determination) {
			if (batch_grade < stop_condition) {
				num_restarts++;
				cout << "No convergence... Retry number " << num_restarts << "!" << endl;
				num_iterations = 0;
			} else {
				cout << "Convergence successful! Finish at epoch " << num_iterations << "!" << endl;
				break;
			}
		} else {
			if (batch_grade < stop_condition) {
				cout << "No convergence..." << endl;
			} else {
				cout << "Convergence successful! Finish at epoch " << num_iterations << "!" << endl;
			}
			break;
		}
	}
	
    /*if (HIDDEN_LAYERS == 0) {
		cout << "Weights for the output layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_4; i++) {
			for (int j = 0; j < INPUTS+1; j++) {
				cout << w4[i][j] << " ";
			}
			cout << endl;
		}
    } else if (HIDDEN_LAYERS == 1) {
		cout << "Weights for the first layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_3; i++) {
			for (int j = 0; j < INPUTS+1; j++) {
				cout << w3[i][j] << " ";
			}
			cout << endl;
		}
		
		cout << "Weights for the output layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_4; i++) {
			for (int j = 0; j < NEURONS_LAYER_3+1; j++) {
				cout << w4[i][j] << " ";
			}
			cout << endl;
		}
    } else if (HIDDEN_LAYERS == 2) {
		cout << "Weights for the first layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_2; i++) {
			for (int j = 0; j < INPUTS+1; j++) {
				cout << w2[i][j] << " ";
			}
			cout << endl;
		}
		
		cout << "Weights for the second layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_3; i++) {
			for (int j = 0; j < NEURONS_LAYER_2+1; j++) {
				cout << w3[i][j] << " ";
			}
			cout << endl;
		}
		
		cout << "Weights for the output layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_4; i++) {
			for (int j = 0; j < NEURONS_LAYER_3+1; j++) {
				cout << w4[i][j] << " ";
			}
			cout << endl;
		}
    } else if (HIDDEN_LAYERS == 3) {
		cout << "Weights for the first layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_1; i++) {
			for (int j = 0; j < INPUTS+1; j++) {
				cout << w1[i][j] << " ";
			}
			cout << endl;
		}
		
		cout << "Weights for the second layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_2; i++) {
			for (int j = 0; j < NEURONS_LAYER_1+1; j++) {
				cout << w2[i][j] << " ";
			}
			cout << endl;
		}
		
		cout << "Weights for the third layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_3; i++) {
			for (int j = 0; j < NEURONS_LAYER_2+1; j++) {
				cout << w3[i][j] << " ";
			}
			cout << endl;
		}
		
		cout << "Weights for the output layer:" << endl;
		for (int i = 0; i < NEURONS_LAYER_4; i++) {
			for (int j = 0; j < NEURONS_LAYER_3+1; j++) {
				cout << w4[i][j] << " ";
			}
			cout << endl;
		}
    }*/
    
    //Testing
    batch_grade = 0;
    for (int i = 0; i < test_size; i++) {
		//Updating input array
		for (int j = 0; j < INPUTS; j++) {
			input[j] = test_data[i][j];
		}
		//Updating reference array (for the future)
		for (int j = 0; j < NEURONS_LAYER_4; j++) {
			reference[j] = test_data[i][INPUTS+j];
		}
		
		//Showing inputs and expected outputs:
		cout << "Inputs:" << endl;
		for (int j = 0; j < INPUTS; j++) {
			cout << input[j] << " ";
		}
		cout << endl;
		cout << "Expected output:" << endl;
		print_disease(reference);
		
		if (HIDDEN_LAYERS == 0) {
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z4[j] = z4[j] + w4[j][k+1]*input[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
				error[j] = reference[j]-s[j];
				batch_grade = batch_grade + pow(error[j],2);
			}
		} else if (HIDDEN_LAYERS == 1) {
			for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
				z3[j] = -w3[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z3[j] = z3[j] + w3[j][k+1]*input[k];
				}
				y3[j] = fatv(z3[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < NEURONS_LAYER_3; k++) {
					z4[j] = z4[j] + w4[j][k+1]*y3[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
				error[j] = reference[j]-s[j];
				batch_grade = batch_grade + pow(error[j],2);
			}
		} else if (HIDDEN_LAYERS == 2) {
			for (int j = 0; j < NEURONS_LAYER_2; j++) { //Frontpropagation
				z2[j] = -w2[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z2[j] = z2[j] + w2[j][k+1]*input[k];
				}
				y2[j] = fatv(z2[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
				z3[j] = -w3[j][0];
				for (int k = 0; k < NEURONS_LAYER_2; k++) {
					z3[j] = z3[j] + w3[j][k+1]*y2[k];
				}
				y3[j] = fatv(z3[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < NEURONS_LAYER_3; k++) {
					z4[j] = z4[j] + w4[j][k+1]*y3[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
				error[j] = reference[j]-s[j];
				batch_grade = batch_grade + pow(error[j],2);
			}
		} else if (HIDDEN_LAYERS == 3) {
			for (int j = 0; j < NEURONS_LAYER_1; j++) { //Frontpropagation
				z1[j] = -w1[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z1[j] = z1[j] + w1[j][k+1]*input[k];
				}
				y1[j] = fatv(z1[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_2; j++) { //Frontpropagation
				z2[j] = -w2[j][0];
				for (int k = 0; k < NEURONS_LAYER_1; k++) {
					z2[j] = z2[j] + w2[j][k+1]*y1[k];
				}
				y2[j] = fatv(z2[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
				z3[j] = -w3[j][0];
				for (int k = 0; k < NEURONS_LAYER_2; k++) {
					z3[j] = z3[j] + w3[j][k+1]*y2[k];
				}
				y3[j] = fatv(z3[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < NEURONS_LAYER_3; k++) {
					z4[j] = z4[j] + w4[j][k+1]*y3[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
				error[j] = reference[j]-s[j];
				batch_grade = batch_grade + pow(error[j],2);
			}
		}
	
		cout << "Output:" << endl;
		print_disease(s);
	} //End testing
	
	mse = (batch_grade/(sample_size*NEURONS_LAYER_4));
	cout << "MSE:" << mse << endl;
    
    cout << "Time to test!" << endl;
    while (true) {
		cout << "What do you feel? Answer with '1' for 'yes' and '0' for 'no'." << endl;
		cout << "Are you coughing?" << endl;
		cin >> input[0];
		cout << "Does your body hurts?" << endl;
		cin >> input[1];
		cout << "Is there a wheezing sound when you breathe?" << endl;
		cin >> input[2];
		cout << "Do you find it hard to breathe?" << endl;
		cin >> input[3];
		cout << "Are you expectorating more mucus than usual?" << endl;
		cin >> input[4];
		cout << "Do you feel a constant pain in the same region of your chest?" << endl;
		cin >> input[5];
		cout << "Do you feel fatigued or exausted?" << endl;
		cin >> input[6];
		cout << "Does your throat hurts?" << endl;
		cin >> input[7];
		cout << "Do you have a fever?" << endl;
		cin >> input[8];
		cout << "Are those symptoms temporary? Do they come and go at random intervals?" << endl;
		cin >> input[9];
		cout << "Do you have a headache?" << endl;
		cin >> input[10];
		cout << "Are you nasal cavities inflammated?" << endl;
		cin >> input[11];
		cout << "Is this going on for a long time already?" << endl;
		cin >> input[12];
		cout << "Is your nose obstructed?" << endl;
		cin >> input[13];
		cout << "Are you sneezing?" << endl;
		cin >> input[14];
		cout << "Do you feel confused anyhow?" << endl;
		cin >> input[15];
		cout << "Do you feel constant chills?" << endl;
		cin >> input[16];
		cout << "Did you cough blood?" << endl;
		cin >> input[17];
		cout << "Is your face sore?" << endl;
		cin >> input[18];
		cout << "Have you been losing weight for no reason?" << endl;
		cin >> input[19];
		cout << "Are you sweating at night?" << endl;
		cin >> input[20];
		cout << "I see... I believe you have... ";
		if (HIDDEN_LAYERS == 0) {
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z4[j] = z4[j] + w4[j][k+1]*input[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
			}
		} else if (HIDDEN_LAYERS == 1) {
			for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
				z3[j] = -w3[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z3[j] = z3[j] + w3[j][k+1]*input[k];
				}
				y3[j] = fatv(z3[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < NEURONS_LAYER_3; k++) {
					z4[j] = z4[j] + w4[j][k+1]*y3[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
			}
		} else if (HIDDEN_LAYERS == 2) {
			for (int j = 0; j < NEURONS_LAYER_2; j++) { //Frontpropagation
				z2[j] = -w2[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z2[j] = z2[j] + w2[j][k+1]*input[k];
				}
				y2[j] = fatv(z2[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
				z3[j] = -w3[j][0];
				for (int k = 0; k < NEURONS_LAYER_2; k++) {
					z3[j] = z3[j] + w3[j][k+1]*y2[k];
				}
				y3[j] = fatv(z3[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < NEURONS_LAYER_3; k++) {
					z4[j] = z4[j] + w4[j][k+1]*y3[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
			}
		} else if (HIDDEN_LAYERS == 3) {
			for (int j = 0; j < NEURONS_LAYER_1; j++) { //Frontpropagation
				z1[j] = -w1[j][0];
				for (int k = 0; k < INPUTS; k++) {
					z1[j] = z1[j] + w1[j][k+1]*input[k];
				}
				y1[j] = fatv(z1[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_2; j++) { //Frontpropagation
				z2[j] = -w2[j][0];
				for (int k = 0; k < NEURONS_LAYER_1; k++) {
					z2[j] = z2[j] + w2[j][k+1]*y1[k];
				}
				y2[j] = fatv(z2[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_3; j++) { //Frontpropagation
				z3[j] = -w3[j][0];
				for (int k = 0; k < NEURONS_LAYER_2; k++) {
					z3[j] = z3[j] + w3[j][k+1]*y2[k];
				}
				y3[j] = fatv(z3[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Frontpropagation
				z4[j] = -w4[j][0];
				for (int k = 0; k < NEURONS_LAYER_3; k++) {
					z4[j] = z4[j] + w4[j][k+1]*y3[k];
				}
				y4[j] = fatv(z4[j]);
			}
			for (int j = 0; j < NEURONS_LAYER_4; j++) { //Gradient calculation
				s[j] = y4[j];
			}
		}
		print_disease(s);
	}
    
    return 0;
}
