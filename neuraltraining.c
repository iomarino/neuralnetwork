#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int max(int* v, size_t n) {
	int max = v[1];
	size_t i;
	for(i=2;i<n;i++) {
		if(v[i]>max)
			max = v[i];
	}
	return max;
}
void shuffle(size_t* v, size_t n) {
    if (n>1) {
        size_t i;
        for(i=0;i<n-1; i++) {
          size_t j = i+rand()/(RAND_MAX/(n-i)+1);
          int t = v[j];
          v[j] = v[i];
          v[i] = t;
        }
    }
}

float sig(float z) {
	return 1/(1+exp(-z));
}
float dsig(float z) {
	return sig(z)*(1-sig(z));
}
float invsig(float z) {
	return log(z/(1-z));
}

int main() {
	srand(time(0));
	int i,j,k,l,r;
	int numLayers = 3;
	int numNeurons[numLayers];
	numNeurons[0] = 2; numNeurons[1] = 2; numNeurons[2] = 1;
	int size = max(numNeurons,numLayers);
	// Generate random biases
	float b[numLayers][size];
	for(i=1;i<numLayers;i++) {
		for(j=0;j<numNeurons[i];j++) {
			b[i][j] = 10*(float)rand()/RAND_MAX-5;
		}
	}
	// Generate random weights
	float w[numLayers][size][size];
	for(i=1;i<numLayers;i++) {
		for(j=0;j<numNeurons[i];j++) {
			for(k=0;k<numNeurons[i-1];k++) {
				w[i][j][k] = 10*(float)rand()/RAND_MAX-5;
			}
		}
	}
	// Generate training inputs and outputs
	size_t numTraining = 10000;
	float x[numTraining][numNeurons[0]];
	float y[numTraining][numNeurons[numLayers-1]];
	for(i=0;i<numTraining;i++) {
		x[i][0] = round((float)rand()/RAND_MAX);
		x[i][1] = round((float)rand()/RAND_MAX);
		if(x[i][0]&&x[i][1])
			y[i][0] = sig(0);
		else if(x[i][0]||x[i][1])
			y[i][0] = sig(1);
		else y[i][0] = sig(0);
		x[i][0] = sig(x[i][0]);
		x[i][1] = sig(x[i][1]);
	}
	// Train the network
	float z[numLayers][size];
	float a[numLayers][size];
	float dCa[numLayers][size];
	float gradCb[numLayers][size];
	float gradCw[numLayers][size][size];
	for(i=1;i<numLayers;i++) {
		for(j=0;j<numNeurons[i];j++) {
			gradCb[i][j] = 0;
			for(k=0;k<numNeurons[i-1];k++) {
				gradCw[i][j][k] = 0;
			}
		}
	}
	float eta = 0.1;
	int epochs = 5;
	int batch_size = 50;
	float cost = 0;
	size_t index[numTraining];
	for(i=0;i<numTraining;i++) {
		index[i] = i;
	}
	FILE* fp = fopen("data","w");
	for(l=0;l<epochs;l++) {
		shuffle(index,numTraining);
		for(r=0;r<numTraining;r++) {
			// Give training input to input neurons
			for(j=0;j<numNeurons[0];j++) {
				a[0][j] = x[index[r]][j];
			}
			// Generate neurons' activations up to last layer
			for(i=1;i<numLayers;i++) {
				for(j=0;j<numNeurons[i];j++) {
					z[i][j] = b[i][j];
					for(k=0;k<numNeurons[i-1];k++) {
						z[i][j] += w[i][j][k]*a[i-1][k];
					}
					a[i][j] = sig(z[i][j]);
				}
			}
			for(j=0;j<numNeurons[numLayers-1];j++) {
				cost += pow(a[numLayers-1][j]-y[index[r]][j],2)/2;
			}
			// Calculate derivatives pertaining the last layer
			for(j=0;j<numNeurons[numLayers-1];j++) {
				dCa[numLayers-1][j] = a[numLayers-1][j]-y[index[r]][j];
				gradCb[numLayers-1][j] += dsig(z[numLayers-1][j])*dCa[numLayers-1][j];
				for(k=0;k<numNeurons[numLayers-2];k++) {
					gradCw[numLayers-1][j][k] += a[numLayers-2][k]*dsig(z[numLayers-1][j])*dCa[numLayers-1][j];
				}
			}
			// Backpropagation
			for(i=numLayers-2;i>0;i--) {
				for(j=0;j<numNeurons[i];j++) {
					dCa[i][j] = 0;
					for(k=0;k<numNeurons[i+1];k++) {
						dCa[i][j] += w[i+1][k][j]*dsig(z[i+1][k])*dCa[i+1][k];
					}
					gradCb[i][j] += dsig(z[i][j])*dCa[i][j];
					for(k=0;k<numNeurons[i-1];k++) {
						gradCw[i][j][k] += a[i-1][k]*dsig(z[i][j])*dCa[i][j];
					}
				}
			}
			// Apply derivatives
			if((r+1)%batch_size==0) {
				for(i=1;i<numLayers;i++) {
					for(j=0;j<numNeurons[i];j++) {
						b[i][j] -= eta*(float)gradCb[i][j]/batch_size;
						for(k=0;k<numNeurons[i-1];k++) {
							w[i][j][k] -= eta*(float)gradCw[i][j][k]/batch_size;
						}
					}
				}
				for(i=1;i<numLayers;i++) {
					for(j=0;j<numNeurons[i];j++) {
						gradCb[i][j] = 0;
						for(k=0;k<numNeurons[i-1];k++) {
							gradCw[i][j][k] = 0;
						}
					}
				}
				fprintf(fp,"%f\n",(float)cost/batch_size);
				cost = 0;
			}
		}
	}
	return 1;
}
