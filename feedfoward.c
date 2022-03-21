#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void read_csv(int row, int col, char *filename, long double **data) {
	FILE *file;
	file = fopen(filename, "r");
	int i = 0;
  char line[4098];
	while (fgets(line, 4098, file) && (i<row)) {
		char* tmp = strdup(line);
		int j = 0;
		const char* tok;
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",")) {
			data[i][j] = atof(tok);
		}
		free(tmp);
		i++;
    }
}
void sample(size_t* v, size_t n, size_t r) {
	for(int i=0; i<r; i++) {
    size_t a = (int)((long double)rand()/RAND_MAX*n);
    v[i] = a;
  }
}
long double stdnorm() {
	long double u1 = (long double)rand()/RAND_MAX;
	long double u2 = (long double)rand()/RAND_MAX;
	return sqrt(-2*log(u1))*cos(2*M_PI*u2);
}
long double sig(long double z) {
	return 1/(1+exp(-z));
}
long double dsig(long double z) {
	return sig(z)*(1-sig(z));
}
long double relu(long double z) {
  if(z>0)
    return z;
  else
    return 0;
}
long double drelu(long double z) {
  if(z>0)
    return 1;
  else
    return 0;
}
long double datan(long double z) {
  return 1/(1+z*z);
}
long double dtanh(long double z) {
  return 1-tanh(z)*tanh(z);
}
// hidden layers activation function
long double g(long double z) { return relu(z); }
long double dg(long double z) { return drelu(z); }
// output layers activation function (not used; softmax implemented in code directly)
long double h(double z) { return sig(z); }
long double dh(double z) { return dsig(z); }

int main() {
	srand(time(0));
  int i, j, k, l, e;
  const int L = 6; // # of layers
  const int K[L] = {784, 256, 128, 64, 32, 10}; // # of neurons per layer
	// tensors declaration
  long double ** z = (long double **)malloc(L*sizeof(long double**));
  for (l=1; l<L; l++) {
    z[l] = (long double *) malloc(K[l]*sizeof(long double));
  }
  long double ** b = (long double **)malloc(L*sizeof(long double**));
  for (l=1; l<L; l++) {
    b[l] = (long double *) malloc(K[l]*sizeof(long double));
  }
  long double *** w = (long double ***)malloc(L*sizeof(long double**));
  for (l=1; l<L; l++) {
    w[l] = (long double **) malloc(K[l]*sizeof(long double *));
    for (k=0; k<K[l]; k++) {
      w[l][k] = (long double *)malloc(K[l-1]*sizeof(long double));
    }
  }
  long double ** a = (long double **)malloc(L*sizeof(long double**));
  for (l=0; l<L; l++) {
    a[l] = (long double *) malloc(K[l]*sizeof(long double));
  }
  long double ** dCa = (long double **)malloc(L*sizeof(long double**));
  for (l=1; l<L; l++) {
    dCa[l] = (long double *) malloc(K[l]*sizeof(long double));
  }
  long double ** dCb = (long double **)malloc(L*sizeof(long double**));
  for (l=1; l<L; l++) {
    dCb[l] = (long double *) malloc(K[l]*sizeof(long double));
  }
  long double *** dCw = (long double ***)malloc(L*sizeof(long double**));
  for (l=1; l<L; l++) {
    dCw[l] = (long double **) malloc(K[l]*sizeof(long double *));
    for (k=0; k<K[l]; k++) {
      dCw[l][k] = (long double *)malloc(K[l-1]*sizeof(long double));
    }
  }

  // source train inputs and outputs
  unsigned int N = 60000; // dimension of train set
  long double ** x = (long double **)malloc(N*sizeof(long double**));
  for (i=0; i<N; i++) {
    x[i] = (long double *)malloc(K[0]*sizeof(long double));
  }
  long double ** y = (long double **)malloc(N*sizeof(long double**));
  for (i=0; i<N; i++) {
    y[i] = (long double *)malloc(K[L-1]*sizeof(long double));
  }
	int row = N;
	int col = K[0]+1;
	long double ** data = (long double **)malloc(row*sizeof(long double *));
	for (i=0; i<row; i++){
		data[i] = (long double *)malloc(col*sizeof(long double));
	}
	read_csv(row, col, "traindata.csv", data);
  for(i=0; i<N; i++) {
    for(j=0; j<K[L-1]; j++) {
      y[i][j] = 0;
      if(data[i][0] == j)
        y[i][j] = 1;
    }
    for(j=0; j<K[0]; j++) {
      x[i][j] = data[i][j+1];
    }
  }
	// source test inputs
  unsigned int N_test = 10000; // dimension of test set
  long double ** x_test = (long double **)malloc(N*sizeof(long double**));
  for (i=0; i<N_test; i++) {
    x_test[i] = (long double *)malloc(K[0]*sizeof(long double));
  }
	row = N_test;
	col = K[0]+1;
  data = realloc(data, row*sizeof(long double *));
	for (i=0; i<row; i++){
		data[i] = (long double *)malloc(col*sizeof(long double));
	}
	read_csv(row, col, "testdata.csv", data);
  for(i=0; i<N_test; i++) {
    for(j=0; j<K[0]; j++) {
      x_test[i][j] = data[i][j+1];
    }
  }
	// training
	// generate random weights and biases
  for(l=1; l<L; l++) {
    for(k=0; k<K[l]; k++) {
      b[l][k] = 2*(long double)rand()/RAND_MAX-1;
      for(j=0; j<K[l-1]; j++) {
        w[l][k][j] = (2*(long double)rand()/RAND_MAX-1)/sqrt(K[l-1]);
      }
    }
  }
	// hyperparameters
	long double eta = 0.008;
	long double lambda = 6;
  unsigned int sgd_size = 30;
	unsigned int steps = 10000;
  size_t index[sgd_size];
  // train
  for(e=0; e<steps; e++) {
		// random sample for SGD
    sample(index, N, sgd_size);
    // reset gradient
    for(l=1; l<L; l++) {
      for(k=0; k<K[l]; k++) {
        dCb[l][k] = 0;
        for(j=0; j<K[l-1]; j++) {
          dCw[l][k][j] = 0;
        }
      }
    }
    // start SGD
    for(i=0; i<sgd_size; i++) {
      // assign activation values for the first layer (input layer)
      for(k=0; k<K[0]; k++) {
        a[0][k] = x[index[i]][k];
      }
      // calculate activation values for the remaining layers
      for(l=1; l<L-1; l++) {
        for(k=0; k<K[l]; k++) {
          z[l][k] = b[l][k];
          for(j=0; j<K[l-1]; j++) {
            z[l][k] += w[l][k][j]*a[l-1][j];
          }
          a[l][k] = g(z[l][k]);
        }
      }
      // softmax for output layer
			long double expsum = 0;
      for(k=0; k<K[L-1]; k++) {
        z[L-1][k] = b[L-1][k];
        for(j=0; j<K[L-2]; j++) {
          z[L-1][k] += w[L-1][k][j]*a[L-2][j];
        }
				expsum += exp(z[L-1][k]);
      }
			for(k=0; k<K[L-1]; k++) {
				a[L-1][k] = exp(z[L-1][k])/expsum;
			}
      // calculate partial derivatives
      for(k=0; k<K[L-1]; k++) {
        dCa[L-1][k] = a[L-1][k]-y[index[i]][k];
        dCb[L-1][k] += dCa[L-1][k];
        for(j=0; j<K[L-2]; j++) {
          dCw[L-1][k][j] += a[L-2][j]*dCa[L-1][k]-(lambda/N)*w[L-1][k][j];
        }
      }
      for(l=L-2; l>0; l--) {
        for(k=0; k<K[l]; k++) {
          dCa[l][k] = 0;
          for(j=0; j<K[l+1]; j++) {
            dCa[l][k] += w[l+1][j][k]*dg(z[l+1][j])*dCa[l+1][j];
          }
          dCb[l][k] += dg(z[l][k])*dCa[l][k];
          for(j=0; j<K[l-1]; j++) {
            dCw[l][k][j] += a[l-1][j]*dg(z[l][k])*dCa[l][k]-(lambda/N)*w[l][k][j];
          }
        }
      }
    }
    // gradient descent
    for(l=1; l<L; l++) {
      for(k=0; k<K[l]; k++) {
        b[l][k] += -eta*dCb[l][k]/sgd_size;
        for(j=0; j<K[l-1]; j++) {
          w[l][k][j] += -eta*dCw[l][k][j]/sgd_size;
        }
      }
    }
  }
  // testing
	FILE* fp = fopen("predictions", "w");
  // calculate predictions on test set
  for(i=0; i<N_test; i++) {
    for(k=0; k<K[0]; k++) {
      a[0][k] = x_test[i][k];
    }
    for(l=1; l<L-1; l++) {
      for(k=0; k<K[l]; k++) {
        z[l][k] = b[l][k];
        for(j=0; j<K[l-1]; j++) {
          z[l][k] += w[l][k][j]*a[l-1][j];
        }
        a[l][k] = g(z[l][k]);
      }
    }
		// exception for output layer
		long double expsum = 0;
		for(k=0; k<K[L-1]; k++) {
			z[L-1][k] = b[L-1][k];
			for(j=0; j<K[L-2]; j++) {
				z[L-1][k] += w[L-1][k][j]*a[L-2][j];
			}
			expsum += exp(z[L-1][k]);
		}
		for(k=0; k<K[L-1]; k++) {
			a[L-1][k] = exp(z[L-1][k])/expsum;
			fprintf(fp, "%Lf,", a[L-1][k]);
		}
    fprintf(fp, "\n");
  }
  fclose(fp);
  // output parameters to file
  fp = fopen("parameters", "w");
  for(l=1; l<L; l++) {
    for(k=0; k<K[l]; k++) {
      fprintf(fp, "%Lf\n", b[l][k]);
      for(j=0; j<K[l-1]; j++) {
        fprintf(fp, "%Lf ", w[l][k][j]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
  }
	fclose(fp);
}
